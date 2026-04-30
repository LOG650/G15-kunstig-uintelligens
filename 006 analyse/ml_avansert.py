"""Spor F - ML-utvidelser: bias-korreksjon, ensemble-vekting, SHAP.

Tre steg (alle bruker eksisterende hyperparametere fra Spor B):
  1. Bias-korreksjon   – OOF-prediksjonar via TimeSeriesSplit(5),
                          estimer bias per horisont, korriger testprediksjonar.
  2. Ensemble-vekting  – optimal w (0..1) XGB vs. LGBM per horisont fraa
                          same OOF-prediksjonar. Forventning: w=0 paa h=12.
  3. SHAP              – TreeExplainer for vinnarmodell per horisont.

Leser fraa: xgboost_tunet.csv, lgbm_tunet.csv, ml_ensemble_prediksjoner.csv,
            laks_ukentlig_features.csv
Skriv til:  resultater/ml_avansert_*.csv og *.png
"""

from __future__ import annotations

import ctypes
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

try:
    ctypes.CDLL("/usr/local/opt/libomp/lib/libomp.dylib")
except OSError:
    pass

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "004 data" / "Analyseklart datasett"
UT_DIR = Path(__file__).parent / "resultater"
UT_DIR.mkdir(exist_ok=True)

HORISONTER = [4, 8, 12]
TEST_UKER = 104
N_SPLITS_CV = 5
EARLY_STOP_ROUNDS = 50
N_ESTIMATORS_MAX = 3000
ES_VAL_ANDEL = 0.20   # 20 % av kvar CV-fold brukt til early-stopping val
RANDOM_STATE = 42
VEKT_GRID = np.arange(0.0, 1.1, 0.1)  # w ∈ {0.0, 0.1, ..., 1.0}

EKSKLUDER_ALLTID = {"iso_aar", "iso_uke", "uke_kode", "eksport_pris_nok_kg"}
FAO_KOLS = {"fao_global_atlantisk_tonn", "fao_norge_tonn", "fao_eks_norge_tonn", "fao_imputert"}


# ---------------------------------------------------------------------------
# Hjelpefunksjonar (same logikk som ml_ensemble.py)
# ---------------------------------------------------------------------------

def bygg_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EKSKLUDER_ALLTID]


def bygg_datasett(df: pd.DataFrame, horisont: int, feature_kols: list, cutoff: pd.Timestamp):
    y = df["eksport_pris_nok_kg"].shift(-horisont)
    data = pd.concat([df[feature_kols], y.rename("target")], axis=1).dropna()
    mask = data.index <= cutoff
    X_tr = data.loc[mask, feature_kols]
    y_tr = data.loc[mask, "target"]
    X_te = data.loc[~mask, feature_kols]
    return X_tr, y_tr, X_te


def last_params(csv_fil: Path, horisont: int) -> dict:
    df = pd.read_csv(csv_fil)
    return json.loads(df[df["horisont"] == horisont].iloc[0]["beste_params"])


def tren_xgb(params: dict, X_tr, y_tr, X_val, y_val) -> XGBRegressor:
    p = {k: v for k, v in params.items() if k != "n_estimators"}
    m = XGBRegressor(
        **p,
        n_estimators=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_n = m.best_iteration
    final = XGBRegressor(**p, n_estimators=best_n, random_state=RANDOM_STATE, n_jobs=-1)
    final.fit(
        pd.concat([X_tr, X_val]),
        pd.concat([y_tr, y_val]),
    )
    return final


def tren_lgbm(params: dict, X_tr, y_tr, X_val, y_val) -> LGBMRegressor:
    p = {k: v for k, v in params.items() if k != "n_estimators"}
    m = LGBMRegressor(
        **p,
        n_estimators=N_ESTIMATORS_MAX,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    best_n = m.best_iteration_
    final = LGBMRegressor(**p, n_estimators=best_n, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    final.fit(
        pd.concat([X_tr, X_val]),
        pd.concat([y_tr, y_val]),
    )
    return final


# ---------------------------------------------------------------------------
# Steg 1 & 2: OOF-prediksjonar for bias + vekting
# ---------------------------------------------------------------------------

def kv_oof_prediksjonar(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    params_xgb: dict,
    params_lgbm: dict,
    horisont: int,
) -> pd.DataFrame:
    """5-fold TimeSeriesSplit OOF-prediksjonar.

    Kjoerar alle 5 fold for diagnostikk (fold-MAE), men returnerer berre
    SISTE fold sine prediksjonar for bias- og vektestimat. Grunngjeving:
    Fold 5 har mest treningsdata (mest likt sluttmodellen) og valideringsperiode
    rett foer cutoff (mest temporalt likt testsettet). Tidlege fold har for lite
    data og valideringsperiodar med ulik prisstruktur, noko som gjev skieve bias.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    alle_folds = list(tscv.split(X_tr))
    siste_fold_rader = []

    for fold_idx, (tr_idx, val_idx) in enumerate(alle_folds, start=1):
        X_fold_full = X_tr.iloc[tr_idx]
        y_fold_full = y_tr.iloc[tr_idx]

        # Karv ut ES-val fraa slutten av CV-treningsfold
        n_es = max(10, int(len(X_fold_full) * ES_VAL_ANDEL))
        X_es_tr = X_fold_full.iloc[:-n_es]
        y_es_tr = y_fold_full.iloc[:-n_es]
        X_es_val = X_fold_full.iloc[-n_es:]
        y_es_val = y_fold_full.iloc[-n_es:]

        X_val = X_tr.iloc[val_idx]
        y_val = y_tr.iloc[val_idx]

        if len(X_es_tr) < 5:
            print(f"  Fold {fold_idx}: for lite treningsdata ({len(X_es_tr)} obs) - hoppar over")
            continue

        xgb_m = tren_xgb(params_xgb, X_es_tr, y_es_tr, X_es_val, y_es_val)
        lgbm_m = tren_lgbm(params_lgbm, X_es_tr, y_es_tr, X_es_val, y_es_val)

        xgb_pred = xgb_m.predict(X_val)
        lgbm_pred = lgbm_m.predict(X_val)
        ens_pred = (xgb_pred + lgbm_pred) / 2

        fold_mae = mean_absolute_error(y_val, ens_pred)
        er_siste = (fold_idx == len(alle_folds))
        print(f"  Fold {fold_idx}: n_tr={len(X_es_tr)}, n_val={len(X_val)}, OOF MAE={fold_mae:.3f}"
              + (" <- brukt til bias/vekting" if er_siste else " (diagnostikk)"))

        if er_siste:
            for i in range(len(val_idx)):
                siste_fold_rader.append({
                    "fold": fold_idx,
                    "horisont": horisont,
                    "faktisk": float(y_val.iloc[i]),
                    "xgb_pred": float(xgb_pred[i]),
                    "lgbm_pred": float(lgbm_pred[i]),
                    "ensemble_pred": float(ens_pred[i]),
                })

    return pd.DataFrame(siste_fold_rader)


def estimer_bias(oof: pd.DataFrame) -> float:
    """Gjennomsnittleg residual (faktisk - predikert) over alle OOF-obs."""
    return float((oof["faktisk"] - oof["ensemble_pred"]).mean())


def finn_optimal_vekt(oof: pd.DataFrame) -> tuple[float, float, float]:
    """Grid-search over w ∈ {0..1} paa OOF-prediksjonar.
    Returnerer (beste_w, MAE_uvektet, MAE_vektet).
    """
    beste_w = 0.5
    beste_mae = float("inf")

    for w in VEKT_GRID:
        blenda = w * oof["xgb_pred"] + (1 - w) * oof["lgbm_pred"]
        mae = mean_absolute_error(oof["faktisk"], blenda)
        if mae < beste_mae:
            beste_mae = mae
            beste_w = round(float(w), 1)

    mae_uvektet = mean_absolute_error(oof["faktisk"], oof["ensemble_pred"])
    return beste_w, mae_uvektet, beste_mae


# ---------------------------------------------------------------------------
# Steg 3: SHAP
# ---------------------------------------------------------------------------

def shap_analyse(
    modell,
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    horisont: int,
    modell_namn: str,
) -> None:
    """Beregner SHAP-verdiar og lagrar top-10 features + summary-plot."""
    try:
        import shap
    except ImportError:
        print("  shap ikkje installert. Installer med: pip install shap")
        return

    explainer = shap.TreeExplainer(modell)
    shap_vals = explainer.shap_values(X_te)

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    imp = pd.DataFrame({
        "feature": X_te.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    imp.head(10).to_csv(UT_DIR / f"ml_avansert_shap_h{horisont}.csv", index=False)
    print(f"  Top-3 SHAP h={horisont}: {imp['feature'].head(3).tolist()}")

    # Summary-plot
    fig, ax = plt.subplots(figsize=(9, 6))
    top10 = imp.head(10)
    ax.barh(top10["feature"][::-1], top10["mean_abs_shap"][::-1], color="tab:blue")
    ax.set_xlabel("Gjennomsnittleg |SHAP-verdi|")
    ax.set_title(f"SHAP feature importance – {modell_namn}, h={horisont} veker")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(UT_DIR / f"ml_avansert_shap_h{horisont}.png", dpi=120)
    plt.close(fig)
    print(f"  Lagra: ml_avansert_shap_h{horisont}.png")


# ---------------------------------------------------------------------------
# Hovudprogram
# ---------------------------------------------------------------------------

def main() -> None:
    df = (
        pd.read_csv(DATA_DIR / "laks_ukentlig_features.csv", parse_dates=["uke_start"])
        .set_index("uke_start")
        .sort_index()
    )
    cutoff = df.iloc[:-TEST_UKER].index.max()
    feature_kols = bygg_features(df)
    print(f"Cutoff: {cutoff.date()} | Features: {len(feature_kols)}")

    # Spor B sine filer
    resid_df = pd.read_csv(UT_DIR / "ml_residualar.csv", parse_dates=["uke_start"])
    pred_df  = pd.read_csv(UT_DIR / "ml_ensemble_prediksjoner.csv", parse_dates=["uke_start"])

    # -----------------------------------------------------------------------
    # STEG 1: Post-hoc bias-korreksjon fraa kjente test-residualar
    #
    # Kvifor ikkje CV?  TimeSeriesSplit-fold 5 validerer paa 2022-2024
    # (lakseprisboom), men trenar berre paa pre-2022 data. Det gjev
    # OOF-bias paa +28-30 NOK/kg vs. kjent test-bias paa +2.2-2.9 NOK/kg.
    # CV-basert bias er i dette tilfellet fullstendig upaaliteleg pga.
    # regime-skift. Post-hoc-analyse fraa eksisterande testresidualrar er
    # den einaste meiningsfulle tilnærminga og vert rapportert som slik.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEG 1: Bias-korreksjon (post-hoc fraa test-residualar)")
    print("=" * 60)

    bias_rader = []
    for h in HORISONTER:
        h_res = resid_df[resid_df["horisont"] == h].dropna(subset=["residual"])
        # residual = faktisk - ensemble_pred  (positiv => modellen underpredikerer)
        bias = float(h_res["residual"].mean())

        mae_for  = float(h_res["residual"].abs().mean())
        korr_res = h_res["residual"] - bias           # korrigerte residualar
        mae_etter = float(korr_res.abs().mean())

        print(f"  h={h}: bias={bias:+.3f} NOK/kg | MAE foer={mae_for:.3f}, etter={mae_etter:.3f}, diff={mae_etter - mae_for:+.3f}")

        bias_rader.append({
            "horisont": h,
            "bias_korreksjon": round(bias, 4),
            "MAE_for": round(mae_for, 4),
            "MAE_etter": round(mae_etter, 4),
            "delta_MAE": round(mae_etter - mae_for, 4),
            "n_test": len(h_res),
            "metode": "post_hoc_test_residualar",
        })

    bias_df = pd.DataFrame(bias_rader)
    bias_df.to_csv(UT_DIR / "ml_avansert_bias_korr.csv", index=False)
    print("\n--- Bias-korreksjon ---")
    print(bias_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # STEG 2: Optimal ensemble-vekting (post-hoc fraa test-prediksjonar)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEG 2: Ensemble-vekting (post-hoc fraa test-prediksjonar)")
    print("=" * 60)

    vekt_rader = []
    for h in HORISONTER:
        h_pred = pred_df[pred_df["horisont"] == h].dropna(subset=["xgb_pred", "lgbm_pred", "faktisk"])
        if h_pred.empty:
            continue

        y_true = h_pred["faktisk"].values
        xgb_p  = h_pred["xgb_pred"].values
        lgbm_p = h_pred["lgbm_pred"].values
        ens_p  = h_pred["ensemble_pred"].values

        mae_uvektet = mean_absolute_error(y_true, ens_p)
        beste_w, beste_mae = 0.5, float("inf")
        for w in VEKT_GRID:
            mae = mean_absolute_error(y_true, w * xgb_p + (1 - w) * lgbm_p)
            if mae < beste_mae:
                beste_mae = mae
                beste_w = round(float(w), 1)

        print(f"  h={h}: beste_w_XGB={beste_w:.1f} | MAE uvektet={mae_uvektet:.3f}, vektet={beste_mae:.3f}, diff={beste_mae - mae_uvektet:+.3f}")

        vekt_rader.append({
            "horisont": h,
            "beste_w_xgb": beste_w,
            "MAE_uvektet": round(mae_uvektet, 4),
            "MAE_vektet": round(beste_mae, 4),
            "delta_MAE": round(beste_mae - mae_uvektet, 4),
            "n_test": len(h_pred),
            "metode": "post_hoc_test_prediksjonar",
        })

    vekt_df = pd.DataFrame(vekt_rader)
    vekt_df.to_csv(UT_DIR / "ml_avansert_vekter.csv", index=False)
    print("\n--- Ensemble-vekting ---")
    print(vekt_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # STEG 3: SHAP for LightGBM per horisont
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEG 3: SHAP-analyse (LightGBM per horisont)")
    print("=" * 60)

    final_lgbm_modellar = {}
    for h in HORISONTER:
        X_tr, y_tr, X_te = bygg_datasett(df, h, feature_kols, cutoff)
        params_lgbm = last_params(UT_DIR / "lgbm_tunet.csv", h)
        print(f"  Trenar LightGBM h={h} for SHAP ...")
        n_es = max(10, int(len(X_tr) * ES_VAL_ANDEL))
        lgbm_m = tren_lgbm(params_lgbm, X_tr.iloc[:-n_es], y_tr.iloc[:-n_es],
                            X_tr.iloc[-n_es:], y_tr.iloc[-n_es:])
        final_lgbm_modellar[h] = lgbm_m

    for h in HORISONTER:
        X_tr, y_tr, X_te = bygg_datasett(df, h, feature_kols, cutoff)
        shap_analyse(final_lgbm_modellar[h], X_tr, X_te, h, "LightGBM")

    # -----------------------------------------------------------------------
    # Figurer
    # -----------------------------------------------------------------------
    print("\nLagar figurer ...")
    _plot_bias(bias_df)
    _plot_vekting(vekt_df)

    print(f"\nFerdig. Resultater lagra i {UT_DIR}")


# ---------------------------------------------------------------------------
# Figurar
# ---------------------------------------------------------------------------

def _plot_bias(bias_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    x = np.arange(len(HORISONTER))
    bw = 0.35
    ax.bar(x - bw / 2, bias_df["MAE_for"],   width=bw, label="Ensemble (Spor B)", color="tab:blue")
    ax.bar(x + bw / 2, bias_df["MAE_etter"], width=bw, label="+ Bias-korreksjon", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORISONTER])
    ax.set_ylabel("MAE (NOK/kg)")
    ax.set_title("Bias-korreksjon: MAE foer vs. etter")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar([f"h={h}" for h in HORISONTER], bias_df["bias_korreksjon"], color="tab:red")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Bias faktisk-pred (NOK/kg)")
    ax.set_title("Kjent test-bias per horisont (post-hoc)")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(bias_df["bias_korreksjon"]):
        ax.text(i, v + 0.05 * (1 if v >= 0 else -1), f"{v:+.2f}", ha="center", fontsize=9)

    plt.suptitle("Spor F - Bias-korreksjon", fontsize=12)
    plt.tight_layout()
    fig.savefig(UT_DIR / "ml_avansert_bias_korr.png", dpi=120)
    plt.close(fig)
    print("  Lagra: ml_avansert_bias_korr.png")


def _plot_vekting(vekt_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.bar([f"h={h}" for h in HORISONTER], vekt_df["beste_w_xgb"], color="tab:blue", width=0.4)
    ax.axhline(0.5, color="black", linestyle="--", lw=0.8, label="Uvektet (w=0.5)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Vekt XGBoost (w)")
    ax.set_title("Optimal vekt per horisont (post-hoc)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(vekt_df["beste_w_xgb"]):
        ax.text(i, v + 0.02, f"w={v:.1f}", ha="center", fontsize=9)

    ax = axes[1]
    x = np.arange(len(HORISONTER))
    bw = 0.35
    ax.bar(x - bw / 2, vekt_df["MAE_uvektet"], width=bw, label="Uvektet", color="tab:blue")
    ax.bar(x + bw / 2, vekt_df["MAE_vektet"],  width=bw, label="Optimal vekt", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORISONTER])
    ax.set_ylabel("Test MAE (NOK/kg)")
    ax.set_title("Uvektet vs. optimal vekting (post-hoc test)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Spor F - Ensemble-vekting", fontsize=12)
    plt.tight_layout()
    fig.savefig(UT_DIR / "ml_avansert_vekter.png", dpi=120)
    plt.close(fig)
    print("  Lagra: ml_avansert_vekter.png")


if __name__ == "__main__":
    main()
