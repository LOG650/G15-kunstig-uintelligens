"""ml_ensemble.py – Ensemble XGBoost + LightGBM med early stopping.

Strategi:
  1. Hent beste hyperparametere fra xgboost_tunet.csv og lgbm_tunet.csv
  2. Refitt med early stopping: siste 52 uker av treningssettet brukes som
     valideringssett (ingen lekkasje – testsett er ikke berørt)
  3. Ensemble: uvektet gjennomsnitt av XGBoost- og LightGBM-prediksjoner
  4. Lagre ml_ensemble.csv og prediksjonsplot per horisont
"""

# Laster libomp for XGBoost på macOS (rpath-problem med homebrew libomp)
import ctypes
try:
    ctypes.CDLL("/usr/local/opt/libomp/lib/libomp.dylib")
except OSError:
    pass

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --- Konstanter -----------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "004 data" / "Analyseklart datasett"
UT_DIR = Path(__file__).parent / "resultater"
UT_DIR.mkdir(exist_ok=True)

HORISONTER = [4, 8, 12]
TEST_UKER = 104
EARLY_STOP_VAL_UKER = 52   # siste 52 uker av treningssett som early-stop-val
EARLY_STOP_ROUNDS = 50     # stopp etter 50 runder utan forbetring
N_ESTIMATORS_MAX = 3000    # øvre grense; early stopping bestemmer faktisk antall

RANDOM_STATE = 42
NAIV_MAE = {4: 8.51, 8: 13.04, 12: 16.35}

EKSKLUDER_ALLTID = {"iso_aar", "iso_uke", "uke_kode", "eksport_pris_nok_kg"}
FAO_KOLS = {"fao_global_atlantisk_tonn", "fao_norge_tonn", "fao_eks_norge_tonn", "fao_imputert"}


# --- Hjelpefunksjoner -----------------------------------------------------

def bygg_features(df: pd.DataFrame) -> list:
    # Bruker uten_fao – samme valg som i ml_eksperiment.py
    ekskluder = EKSKLUDER_ALLTID | FAO_KOLS
    return [c for c in df.columns if c not in ekskluder]


def bygg_datasett(df: pd.DataFrame, horisont: int, feature_kols: list, cutoff: pd.Timestamp):
    y = df["eksport_pris_nok_kg"].shift(-horisont)
    data = pd.concat([df[feature_kols], y.rename("target")], axis=1).dropna()
    mask = data.index <= cutoff
    X_tr = data.loc[mask, feature_kols]
    y_tr = data.loc[mask, "target"]
    X_te = data.loc[~mask, feature_kols]
    return X_tr, y_tr, X_te


def prediker(model, X_te: pd.DataFrame, test_index: pd.DatetimeIndex, horisont: int) -> pd.Series:
    pred = pd.Series(model.predict(X_te), index=X_te.index)
    pred.index = pred.index + pd.Timedelta(weeks=horisont)
    return pred.reindex(test_index)


def evaluer(y_true: pd.Series, y_pred: pd.Series, horisont: int) -> dict:
    par = pd.concat([y_true, y_pred], axis=1).dropna()
    return {
        "horisont": horisont,
        "n": len(par) if not par.empty else 0,
        "MAE": mean_absolute_error(par.iloc[:, 0], par.iloc[:, 1]) if not par.empty else np.nan,
        "MAPE": mean_absolute_percentage_error(par.iloc[:, 0], par.iloc[:, 1]) if not par.empty else np.nan,
    }


def last_params(csv_path: Path, horisont: int) -> dict:
    df = pd.read_csv(csv_path)
    row = df[df["horisont"] == horisont].iloc[0]
    return json.loads(row["beste_params"])


# --- Hovudprogram ---------------------------------------------------------

def main() -> None:
    df = (
        pd.read_csv(DATA_DIR / "laks_ukentlig_features.csv", parse_dates=["uke_start"])
        .set_index("uke_start")
        .sort_index()
    )
    test = df.iloc[-TEST_UKER:]
    cutoff = df.iloc[:-TEST_UKER].index.max()
    feature_kols = bygg_features(df)

    print(f"Cutoff: {cutoff.date()} | Test: {test.index.min().date()} – {test.index.max().date()}")
    print(f"Features: {len(feature_kols)} (uten FAO)\n")

    resultater = []
    alle_pred = {}   # {horisont: {"xgb": Series, "lgbm": Series, "ensemble": Series}}

    for h in HORISONTER:
        print(f"{'=' * 60}")
        print(f"Horisont h={h}")
        print(f"{'=' * 60}")

        X_tr_full, y_tr_full, X_te = bygg_datasett(df, h, feature_kols, cutoff)

        # Karv ut early-stopping-valideringssett fra treningssettet
        X_es_tr = X_tr_full.iloc[:-EARLY_STOP_VAL_UKER]
        y_es_tr = y_tr_full.iloc[:-EARLY_STOP_VAL_UKER]
        X_es_val = X_tr_full.iloc[-EARLY_STOP_VAL_UKER:]
        y_es_val = y_tr_full.iloc[-EARLY_STOP_VAL_UKER:]
        print(f"  Treningssett: {len(X_es_tr)} obs | ES-val: {len(X_es_val)} obs | Test: {len(X_te)} obs")

        # --- XGBoost med early stopping ---
        xgb_params = last_params(UT_DIR / "xgboost_tunet.csv", h)
        xgb_params.pop("n_estimators", None)   # overstyres av N_ESTIMATORS_MAX + early stop
        xgb = XGBRegressor(
            **xgb_params,
            n_estimators=N_ESTIMATORS_MAX,
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb.fit(X_es_tr, y_es_tr, eval_set=[(X_es_val, y_es_val)], verbose=False)
        xgb_best_iter = xgb.best_iteration
        print(f"  XGBoost: beste iterasjon = {xgb_best_iter}")

        # Refit på fullt treningssett med det funnet antall iterasjoner
        xgb_final = XGBRegressor(
            **xgb_params,
            n_estimators=xgb_best_iter,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb_final.fit(X_tr_full, y_tr_full)
        pred_xgb = prediker(xgb_final, X_te, test.index, h)

        res_xgb = evaluer(test["eksport_pris_nok_kg"], pred_xgb, h)
        print(f"  XGBoost MAE={res_xgb['MAE']:.2f} | MAPE={res_xgb['MAPE']:.1%}")

        # --- LightGBM med early stopping ---
        lgbm_params = last_params(UT_DIR / "lgbm_tunet.csv", h)
        lgbm_params.pop("n_estimators", None)
        lgbm = LGBMRegressor(
            **lgbm_params,
            n_estimators=N_ESTIMATORS_MAX,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(
            X_es_tr, y_es_tr,
            eval_set=[(X_es_val, y_es_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        lgbm_best_iter = lgbm.best_iteration_
        print(f"  LightGBM: beste iterasjon = {lgbm_best_iter}")

        lgbm_final = LGBMRegressor(
            **lgbm_params,
            n_estimators=lgbm_best_iter,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm_final.fit(X_tr_full, y_tr_full)
        pred_lgbm = prediker(lgbm_final, X_te, test.index, h)

        res_lgbm = evaluer(test["eksport_pris_nok_kg"], pred_lgbm, h)
        print(f"  LightGBM MAE={res_lgbm['MAE']:.2f} | MAPE={res_lgbm['MAPE']:.1%}")

        # --- Ensemble: uvektet gjennomsnitt ---
        pred_ens = (pred_xgb.fillna(0) + pred_lgbm.fillna(0)) / 2
        # Sett NaN der begge er NaN
        begge_nan = pred_xgb.isna() & pred_lgbm.isna()
        pred_ens[begge_nan] = np.nan

        res_ens = evaluer(test["eksport_pris_nok_kg"], pred_ens, h)
        print(f"  Ensemble  MAE={res_ens['MAE']:.2f} | MAPE={res_ens['MAPE']:.1%}")
        print(f"  Naiv      MAE={NAIV_MAE[h]:.2f}")
        slaar = res_ens["MAE"] < NAIV_MAE[h]
        print(f"  Slår naiv: {'JA ✓' if slaar else 'NEI ✗'}\n")

        for res, modell in [(res_xgb, "XGBoost+ES"), (res_lgbm, "LightGBM+ES"), (res_ens, "Ensemble")]:
            res["modell"] = modell
            resultater.append(res)

        alle_pred[h] = {
            "xgb": pred_xgb,
            "lgbm": pred_lgbm,
            "ensemble": pred_ens,
        }

    # --- Lagre CSV --------------------------------------------------------
    res_df = pd.DataFrame(resultater)[["horisont", "modell", "n", "MAE", "MAPE"]]
    res_df.to_csv(UT_DIR / "ml_ensemble.csv", index=False)

    # --- Prediksjonsplot --------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 14), sharex=False)
    for ax, h in zip(axes, HORISONTER):
        faktisk = test["eksport_pris_nok_kg"]
        pred = alle_pred[h]

        ax.plot(faktisk.index, faktisk.values, label="Faktisk pris", color="black", linewidth=1.8)
        ax.plot(faktisk.index, pred["xgb"].values, label="XGBoost+ES", color="#FF5722",
                linestyle="--", linewidth=1.2, alpha=0.8)
        ax.plot(faktisk.index, pred["lgbm"].values, label="LightGBM+ES", color="#4CAF50",
                linestyle="--", linewidth=1.2, alpha=0.8)
        ax.plot(faktisk.index, pred["ensemble"].values, label="Ensemble", color="#2196F3",
                linewidth=1.8)

        ens_mae = res_df.loc[(res_df["horisont"] == h) & (res_df["modell"] == "Ensemble"), "MAE"].values[0]
        naiv_str = f"Naiv MAE={NAIV_MAE[h]:.2f}"
        ens_str = f"Ensemble MAE={ens_mae:.2f}"
        slaar = ens_mae < NAIV_MAE[h]
        ax.set_title(f"h={h} veker | {ens_str} | {naiv_str} | {'✓ slår naiv' if slaar else '✗ slår ikke naiv'}",
                     fontsize=11)
        ax.set_ylabel("NOK/kg")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Dato")
    fig.suptitle("Ensemble XGBoost + LightGBM – testperiode 2024–2026", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(UT_DIR / "ml_ensemble_prediksjon.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Samandreg --------------------------------------------------------
    print("=" * 60)
    print("SAMANDREG")
    print("=" * 60)
    header = f"{'Horisont':>10} | {'Naiv MAE':>9} | {'XGB+ES':>8} | {'LGBM+ES':>9} | {'Ensemble':>9} | {'Slår naiv':>10}"
    print(header)
    print("-" * len(header))
    alle_slaar = True
    for h in HORISONTER:
        xmae = res_df.loc[(res_df["horisont"] == h) & (res_df["modell"] == "XGBoost+ES"), "MAE"].values[0]
        lmae = res_df.loc[(res_df["horisont"] == h) & (res_df["modell"] == "LightGBM+ES"), "MAE"].values[0]
        emae = res_df.loc[(res_df["horisont"] == h) & (res_df["modell"] == "Ensemble"), "MAE"].values[0]
        slaar = emae < NAIV_MAE[h]
        alle_slaar = alle_slaar and slaar
        print(f"  {h:>8}  | {NAIV_MAE[h]:>9.2f} | {xmae:>8.2f} | {lmae:>9.2f} | {emae:>9.2f} | {'JA ✓' if slaar else 'NEI ✗':>10}")
    print(f"\nSuksesskriterie (alle slår naiv): {'JA ✓' if alle_slaar else 'NEI ✗'}")
    print(f"\nResultat lagret til: {UT_DIR}")


if __name__ == "__main__":
    main()
