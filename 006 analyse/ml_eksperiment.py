"""ML-eksperiment Spor B: FAO-sammenligning, XGBoost-tuning, LightGBM, feature importance.

Oppgaver:
  1. FAO-handtering: med vs. uten FAO-kolonner (forward-fill + fao_imputert-flagg)
  2. Hyperparameter-tuning XGBoost (RandomizedSearchCV + TimeSeriesSplit)
  3. LightGBM med same tuning-protokoll
  4. Feature importance for vinnarmodell per horisont (top 15)
"""

# Laster libomp foer xgboost paa macOS (rpath-problem med homebrew libomp)
import ctypes
try:
    ctypes.CDLL("/usr/local/opt/libomp/lib/libomp.dylib")
except OSError:
    pass

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --- Konstanter -----------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "004 data" / "Analyseklart datasett"
UT_DIR = Path(__file__).parent / "resultater"
UT_DIR.mkdir(exist_ok=True)

HORISONTER = [4, 8, 12]
TEST_UKER = 104
N_SPLITS_CV = 5
N_ITER = 60
RANDOM_STATE = 42

EKSKLUDER_ALLTID = {"iso_aar", "iso_uke", "uke_kode", "eksport_pris_nok_kg"}
FAO_KOLS = {"fao_global_atlantisk_tonn", "fao_norge_tonn", "fao_eks_norge_tonn", "fao_imputert"}

# Naiv baseline-MAE (fra baseline_metrikker_pivot.csv) - brukt i samandraget
NAIV_MAE = {4: 8.51, 8: 13.04, 12: 16.35}

# --- Hjelpefunksjonar -----------------------------------------------------

def bygg_features(df: pd.DataFrame, inkluder_fao: bool) -> list:
    ekskluder = EKSKLUDER_ALLTID.copy()
    if not inkluder_fao:
        ekskluder |= FAO_KOLS
    return [c for c in df.columns if c not in ekskluder]


def bygg_datasett(df: pd.DataFrame, horisont: int, feature_kols: list, cutoff: pd.Timestamp):
    """Lag X_tr, y_tr, X_te med same split-logikk som baseline."""
    y = df["eksport_pris_nok_kg"].shift(-horisont)
    data = pd.concat([df[feature_kols], y.rename("target")], axis=1).dropna()
    mask = data.index <= cutoff
    X_tr = data.loc[mask, feature_kols]
    y_tr = data.loc[mask, "target"]
    X_te = data.loc[~mask, feature_kols]
    return X_tr, y_tr, X_te


def prediker(model, X_te: pd.DataFrame, test_index: pd.DatetimeIndex, horisont: int) -> pd.Series:
    """Prediker og juster tidsindeks h veker fram (same som baseline)."""
    pred = pd.Series(model.predict(X_te), index=X_te.index)
    pred.index = pred.index + pd.Timedelta(weeks=horisont)
    return pred.reindex(test_index)


def evaluer(y_true: pd.Series, y_pred: pd.Series, horisont: int, beste_params: dict | None = None) -> dict:
    par = pd.concat([y_true, y_pred], axis=1).dropna()
    res = {
        "horisont": horisont,
        "n": len(par) if not par.empty else 0,
        "MAE": mean_absolute_error(par.iloc[:, 0], par.iloc[:, 1]) if not par.empty else np.nan,
        "MAPE": mean_absolute_percentage_error(par.iloc[:, 0], par.iloc[:, 1]) if not par.empty else np.nan,
    }
    if beste_params is not None:
        res["beste_params"] = json.dumps(beste_params)
    return res


# --- Hovudprogram ---------------------------------------------------------

def main() -> None:
    df = (
        pd.read_csv(DATA_DIR / "laks_ukentlig_features.csv", parse_dates=["uke_start"])
        .set_index("uke_start")
        .sort_index()
    )
    test = df.iloc[-TEST_UKER:]
    cutoff = df.iloc[:-TEST_UKER].index.max()
    print(f"Periode: {df.index.min().date()} – {df.index.max().date()}")
    print(f"Cutoff (siste treningsveke): {cutoff.date()}")
    print(f"Test: {test.index.min().date()} – {test.index.max().date()} ({len(test)} veker)\n")

    # -----------------------------------------------------------------
    # 1. FAO-samanlikning (baseline XGBoost-params, med vs. uten FAO)
    # -----------------------------------------------------------------
    print("=" * 60)
    print("1. FAO-samanlikning")
    print("=" * 60)
    xgb_params_basis = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    fao_res = []
    for inkluder_fao, label in [(False, "uten_fao"), (True, "med_fao")]:
        feat = bygg_features(df, inkluder_fao)
        for h in HORISONTER:
            X_tr, y_tr, X_te = bygg_datasett(df, h, feat, cutoff)
            m = XGBRegressor(**xgb_params_basis)
            m.fit(X_tr, y_tr)
            pred = prediker(m, X_te, test.index, h)
            res = evaluer(test["eksport_pris_nok_kg"], pred, h)
            res["variant"] = label
            fao_res.append(res)
            print(f"  h={h:2d} | {label:8s} | n={res['n']:3d} | MAE={res['MAE']:6.2f} | MAPE={res['MAPE']:.1%}")

    fao_df = pd.DataFrame(fao_res)
    fao_df.to_csv(UT_DIR / "ml_fao_sammenligning.csv", index=False)

    # Vel FAO-variant basert paa gjennomsnittleg MAE over alle horisontar
    snitt_mae = fao_df.groupby("variant")["MAE"].mean()
    bruk_fao = snitt_mae.idxmin() == "med_fao"
    print(f"\nSnitt MAE: {snitt_mae.to_dict()}")
    print(f"Vel variant: {'med_fao' if bruk_fao else 'uten_fao'}")
    feature_kols = bygg_features(df, bruk_fao)
    print(f"Antal features i bruk: {len(feature_kols)}")

    # -----------------------------------------------------------------
    # 2. XGBoost hyperparameter-tuning
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. XGBoost hyperparameter-tuning")
    print("=" * 60)
    xgb_param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1.0, 10.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    xgb_resultater = []
    xgb_modeller = {}

    for h in HORISONTER:
        X_tr, y_tr, X_te = bygg_datasett(df, h, feature_kols, cutoff)
        print(f"  h={h}: trenar {len(X_tr)} obs, søk over {N_ITER} kombinasjonar ...", flush=True)
        search = RandomizedSearchCV(
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_distributions=xgb_param_dist,
            n_iter=N_ITER,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        pred = prediker(best, X_te, test.index, h)
        res = evaluer(test["eksport_pris_nok_kg"], pred, h, search.best_params_)
        xgb_resultater.append(res)
        xgb_modeller[h] = best
        print(f"       MAE={res['MAE']:.2f} | MAPE={res['MAPE']:.1%} | beste params={search.best_params_}")

    pd.DataFrame(xgb_resultater).to_csv(UT_DIR / "xgboost_tunet.csv", index=False)

    # -----------------------------------------------------------------
    # 3. LightGBM hyperparameter-tuning
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. LightGBM hyperparameter-tuning")
    print("=" * 60)
    lgbm_param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [-1, 4, 6, 8, 10],
        "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1.0, 10.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 30, 50],
    }
    lgbm_resultater = []
    lgbm_modeller = {}

    for h in HORISONTER:
        X_tr, y_tr, X_te = bygg_datasett(df, h, feature_kols, cutoff)
        print(f"  h={h}: trenar {len(X_tr)} obs, søk over {N_ITER} kombinasjonar ...", flush=True)
        search = RandomizedSearchCV(
            LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            param_distributions=lgbm_param_dist,
            n_iter=N_ITER,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        pred = prediker(best, X_te, test.index, h)
        res = evaluer(test["eksport_pris_nok_kg"], pred, h, search.best_params_)
        lgbm_resultater.append(res)
        lgbm_modeller[h] = best
        print(f"       MAE={res['MAE']:.2f} | MAPE={res['MAPE']:.1%} | beste params={search.best_params_}")

    pd.DataFrame(lgbm_resultater).to_csv(UT_DIR / "lgbm_tunet.csv", index=False)

    # -----------------------------------------------------------------
    # 4. Feature importance – vinnarmodell per horisont (top 15)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. Feature importance")
    print("=" * 60)
    xgb_df = pd.DataFrame(xgb_resultater)
    lgbm_df = pd.DataFrame(lgbm_resultater)
    vinnarar = {}

    for h in HORISONTER:
        xgb_mae = xgb_df.loc[xgb_df["horisont"] == h, "MAE"].values[0]
        lgbm_mae = lgbm_df.loc[lgbm_df["horisont"] == h, "MAE"].values[0]
        if xgb_mae <= lgbm_mae:
            namn = "XGBoost"
            modell = xgb_modeller[h]
        else:
            namn = "LightGBM"
            modell = lgbm_modeller[h]
        vinnarar[h] = namn
        print(f"  h={h}: vinnar={namn} (XGB MAE={xgb_mae:.2f}, LGBM MAE={lgbm_mae:.2f})")

        imp = pd.DataFrame({
            "feature": feature_kols,
            "importance": modell.feature_importances_,
        }).sort_values("importance", ascending=False).head(15).reset_index(drop=True)
        imp.to_csv(UT_DIR / f"ml_feature_importance_h{h}.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#2196F3")
        ax.set_title(f"Feature importance – {namn}, h={h} veker", fontsize=13)
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(UT_DIR / f"ml_feature_importance_h{h}.png", dpi=120)
        plt.close(fig)

    # -----------------------------------------------------------------
    # 5. Samandrags-plot: MAE per horisont, alle modellar
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(HORISONTER))
    w = 0.22
    ax.bar(x - w, [NAIV_MAE[h] for h in HORISONTER], width=w, label="Naiv", color="#9E9E9E")
    ax.bar(x, xgb_df["MAE"].values, width=w, label="XGBoost (tunet)", color="#FF5722")
    ax.bar(x + w, lgbm_df["MAE"].values, width=w, label="LightGBM (tunet)", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORISONTER])
    ax.set_ylabel("MAE (NOK/kg)")
    ax.set_title("Testset MAE per horisont – Spor B")
    ax.legend()
    fig.tight_layout()
    fig.savefig(UT_DIR / "ml_mae_samanlikning.png", dpi=120)
    plt.close(fig)

    # -----------------------------------------------------------------
    # 6. Skriver ut samandreg
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAMANDREG")
    print("=" * 60)
    header = f"{'Horisont':>10} | {'Naiv MAE':>9} | {'XGB tunet':>10} | {'LGBM tunet':>11} | {'Vinnar':>9}"
    print(header)
    print("-" * len(header))
    alle_slaar_naiv = True
    for h in HORISONTER:
        xmae = xgb_df.loc[xgb_df["horisont"] == h, "MAE"].values[0]
        lmae = lgbm_df.loc[lgbm_df["horisont"] == h, "MAE"].values[0]
        best_ml = min(xmae, lmae)
        slaar = best_ml < NAIV_MAE[h]
        alle_slaar_naiv = alle_slaar_naiv and slaar
        print(
            f"  {h:>8}  | {NAIV_MAE[h]:>9.2f} | {xmae:>10.2f} | {lmae:>11.2f} | {vinnarar[h]:>9}"
            f"  {'✓' if slaar else '✗'}"
        )
    print(f"\nSuksesskriterie (alle horisontane slaar naiv): {'JA ✓' if alle_slaar_naiv else 'NEI ✗'}")
    print(f"\nResultat lagra til: {UT_DIR}")


if __name__ == "__main__":
    main()
