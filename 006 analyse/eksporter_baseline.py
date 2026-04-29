"""Kjorer baseline-modellene og lagrer MAE/MAPE-tabellen til CSV.

Speiler logikken i baseline_modeller.ipynb, men uten plotting/visning,
slik at vi har et reproduserbart referansetall pa disk.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "004 data" / "Analyseklart datasett"
UT_DIR = Path(__file__).parent / "resultater"
UT_DIR.mkdir(exist_ok=True)

HORISONTER = [4, 8, 12]
TEST_UKER = 104


def evaluer(y_true: pd.Series, y_pred: pd.Series, modell: str, horisont: int) -> dict:
    par = pd.concat([y_true, y_pred], axis=1).dropna()
    if par.empty:
        return {"modell": modell, "horisont": horisont, "n": 0, "MAE": np.nan, "MAPE": np.nan}
    return {
        "modell": modell,
        "horisont": horisont,
        "n": len(par),
        "MAE": mean_absolute_error(par.iloc[:, 0], par.iloc[:, 1]),
        "MAPE": mean_absolute_percentage_error(par.iloc[:, 0], par.iloc[:, 1]),
    }


def main() -> None:
    df = (
        pd.read_csv(DATA_DIR / "laks_ukentlig_features.csv", parse_dates=["uke_start"])
        .set_index("uke_start")
        .sort_index()
    )
    train = df.iloc[:-TEST_UKER].copy()
    test = df.iloc[-TEST_UKER:].copy()

    resultater: list[dict] = []

    # A) Naiv
    for h in HORISONTER:
        pred = test["eksport_pris_nok_kg"].shift(h)
        resultater.append(evaluer(test["eksport_pris_nok_kg"], pred, "Naiv", h))

    # B) SARIMA (single-fit, multi-step - se notebook for advarsel)
    y_train = train["eksport_pris_nok_kg"].asfreq("W-MON")
    y_test = test["eksport_pris_nok_kg"].asfreq("W-MON")
    sarima = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    forecast_full = sarima.get_forecast(steps=len(y_test)).predicted_mean
    forecast_full.index = y_test.index
    for h in HORISONTER:
        resultater.append(evaluer(y_test, forecast_full, "SARIMA", h))

    # C) XGBoost direkte multi-horisont
    feature_kols = [
        c for c in df.columns
        if c not in {
            "iso_aar", "iso_uke", "uke_kode",
            "eksport_pris_nok_kg",
            "fao_global_atlantisk_tonn", "fao_norge_tonn", "fao_eks_norge_tonn",
        }
    ]
    cutoff = train.index.max()
    for h in HORISONTER:
        y = df["eksport_pris_nok_kg"].shift(-h)
        X = df[feature_kols]
        data = pd.concat([X, y.rename("target")], axis=1).dropna()
        train_mask = data.index <= cutoff
        X_tr, y_tr = data.loc[train_mask, feature_kols], data.loc[train_mask, "target"]
        X_te = data.loc[~train_mask, feature_kols]

        model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        pred = pd.Series(model.predict(X_te), index=X_te.index)
        pred.index = pred.index + pd.Timedelta(weeks=h)
        pred = pred.reindex(test.index)
        resultater.append(evaluer(test["eksport_pris_nok_kg"], pred, "XGBoost", h))

    res = pd.DataFrame(resultater)
    res.to_csv(UT_DIR / "baseline_metrikker.csv", index=False)

    pivot = res.pivot(index="horisont", columns="modell", values=["MAE", "MAPE"])
    pivot = pivot.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    pivot.to_csv(UT_DIR / "baseline_metrikker_pivot.csv")

    print(pivot.round(3).to_string())
    print(f"\nLagret til {UT_DIR}")


if __name__ == "__main__":
    main()
