"""Spor A - SARIMA / SARIMAX med rullende-opphav-evaluering.

Erstatter baseline-notebookens single-fit + multi-step forecast med en walk-forward
prosedyre der modellen ved hvert steg "vet" alt opp til tidspunkt t, forecaster h
skritt fram og sammenlignes mot faktisk pris pa t+h. Refit av params gjores kun pa
treningsdata; under walk-forward brukes results.append(..., refit=False) for fart.

Lagrer metrikker, prognoser med 95% CI, og figurer til resultater/.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "004 data" / "Analyseklart datasett"
UT_DIR = Path(__file__).parent / "resultater"
UT_DIR.mkdir(exist_ok=True)

HORISONTER = [4, 8, 12]
TEST_UKER = 104
EXOG_KOLS = ["eur_nok_snitt", "usd_nok_snitt"]
ALPHA = 0.05  # for 95% konfidensintervall

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 52)


def evaluer(y_true: pd.Series, y_pred: pd.Series, modell: str, horisont: int) -> dict:
    par = pd.concat([y_true.rename("y"), y_pred.rename("yhat")], axis=1).dropna()
    if par.empty:
        return {"modell": modell, "horisont": horisont, "n": 0, "MAE": np.nan, "MAPE": np.nan}
    return {
        "modell": modell,
        "horisont": horisont,
        "n": len(par),
        "MAE": mean_absolute_error(par["y"], par["yhat"]),
        "MAPE": mean_absolute_percentage_error(par["y"], par["yhat"]),
    }


def rullerende_prognose(
    y_train: pd.Series,
    y_test: pd.Series,
    exog_train: pd.DataFrame | None = None,
    exog_test: pd.DataFrame | None = None,
    modell_navn: str = "SARIMA",
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, "SARIMAXResults"]:
    """Walk-forward forecast med fast params (refit=False).

    Returnerer:
      prognoser: dict h -> DataFrame med kolonner [yhat, yhat_low, yhat_high] indeksert pa
                 maaltidspunkt (t+h)
      metrikker: DataFrame med en rad per horisont
      init: SARIMAXResults fra treningsfit (brukt til residualdiagnostikk)
    """
    print(f"\n[{modell_navn}] Tilpasser modell pa {len(y_train)} treningspunkter ...")
    init = SARIMAX(
        y_train,
        exog=exog_train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    print(f"[{modell_navn}] Treningsfit ferdig. AIC={init.aic:.1f}")

    # Init buffere
    prognoser: dict[int, dict[pd.Timestamp, tuple[float, float, float]]] = {
        h: {} for h in HORISONTER
    }
    h_max = max(HORISONTER)

    current = init
    n_test = len(y_test)
    print(f"[{modell_navn}] Walk-forward over {n_test} testuker ...")

    for step in range(n_test):
        # Forecast h_max skritt fram fra na
        if exog_test is not None:
            # Vi trenger eksogene verdier for de neste h_max stegene. I et virkelig
            # produksjonsscenario maatte disse vaert egne prognoser; her bruker vi
            # de faktiske kjente verdiene som best mulig proxy.
            exog_future = exog_test.iloc[step : step + h_max]
            if len(exog_future) < h_max:
                # mot slutten av testperioden; pad med siste verdi
                pad = exog_test.iloc[[-1]]
                exog_future = pd.concat([exog_future] + [pad] * (h_max - len(exog_future)))
                exog_future = exog_future.iloc[:h_max]
            fc = current.get_forecast(steps=h_max, exog=exog_future)
        else:
            fc = current.get_forecast(steps=h_max)

        fc_mean = fc.predicted_mean.values
        fc_ci = fc.conf_int(alpha=ALPHA).values

        for h in HORISONTER:
            target_step = step + h - 1
            if target_step < n_test:
                target_date = y_test.index[target_step]
                prognoser[h][target_date] = (
                    fc_mean[h - 1],
                    fc_ci[h - 1, 0],
                    fc_ci[h - 1, 1],
                )

        # Append obs ved tidspunkt step (slik at neste iter har "kunnskap" t.o.m. step)
        nytt_y = y_test.iloc[step : step + 1]
        nytt_exog = exog_test.iloc[step : step + 1] if exog_test is not None else None
        current = current.append(nytt_y, exog=nytt_exog, refit=False)

        if (step + 1) % 26 == 0 or step == n_test - 1:
            print(f"[{modell_navn}]   {step + 1}/{n_test} uker behandlet")

    # Bygg resultatframer
    prognose_dfs: dict[int, pd.DataFrame] = {}
    metrikker_rader = []
    for h in HORISONTER:
        rows = sorted(prognoser[h].items())
        idx = [d for d, _ in rows]
        vals = np.array([v for _, v in rows])
        df = pd.DataFrame(
            vals, index=pd.DatetimeIndex(idx, name="uke_start"),
            columns=["yhat", "yhat_low", "yhat_high"],
        )
        prognose_dfs[h] = df
        metrikker_rader.append(
            evaluer(y_test.loc[df.index], df["yhat"], modell_navn, h)
        )

    return prognose_dfs, pd.DataFrame(metrikker_rader), init


def ci_dekning(
    prognoser: dict[int, pd.DataFrame],
    y_test: pd.Series,
    modell_navn: str,
    nominell: float = 0.95,
) -> pd.DataFrame:
    """Andel testobservasjoner som faller innenfor det nominelle 95 % intervallet.

    Returnerer DataFrame med kolonner [modell, horisont, n, dekning, gj_bredde].
    Hvis dekning er langt unna 95 % er CI-ene daarlig kalibrerte.
    """
    rader = []
    for h, prog in prognoser.items():
        y = y_test.loc[prog.index]
        innenfor = (y >= prog["yhat_low"]) & (y <= prog["yhat_high"])
        rader.append({
            "modell": modell_navn,
            "horisont": h,
            "nominell": nominell,
            "n": len(y),
            "dekning": innenfor.mean(),
            "gj_bredde": (prog["yhat_high"] - prog["yhat_low"]).mean(),
        })
    return pd.DataFrame(rader)


def residualdiagnostikk(
    init: "SARIMAXResults",
    modell_navn: str,
    ut_dir: Path,
) -> dict:
    """Ljung-Box (autokorr.), Jarque-Bera (normalitet), pluss ACF + QQ-plot.

    Bruker standardiserte in-sample residualer fra treningsfit
    (`standardized_forecasts_error`). Skipper de forste 52 obs for a unngaa
    diffuse-init-effekter fra sesongleddet.
    """
    std_err = init.standardized_forecasts_error[0]  # univariat: ta forste rad
    resid = pd.Series(std_err).iloc[52:].dropna()

    # Ljung-Box pa flere lag
    lb = acorr_ljungbox(resid, lags=[10, 20, 52], return_df=True)
    lb.index.name = "lag"

    # Jarque-Bera (normalitet) - statsmodels-versjonen returnerer (stat, p, skew, kurt)
    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid.values)

    diag = {
        "modell": modell_navn,
        "n_resid": len(resid),
        "ljungbox_lag10_p": float(lb.loc[10, "lb_pvalue"]),
        "ljungbox_lag20_p": float(lb.loc[20, "lb_pvalue"]),
        "ljungbox_lag52_p": float(lb.loc[52, "lb_pvalue"]),
        "jb_stat": float(jb_stat),
        "jb_p": float(jb_p),
        "skew": float(jb_skew),
        "kurtosis": float(jb_kurt),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_acf(resid, lags=52, ax=axes[0])
    axes[0].set_title(f"{modell_navn} - residual ACF")
    sps.probplot(resid.values, dist="norm", plot=axes[1])
    axes[1].set_title(f"{modell_navn} - QQ-plot")
    plt.tight_layout()
    fig.savefig(ut_dir / f"{modell_navn.lower()}_residualer.png", dpi=120)
    plt.close(fig)

    return diag


def plot_prognoser(
    y_test: pd.Series,
    sarima_prog: dict[int, pd.DataFrame],
    sarimax_prog: dict[int, pd.DataFrame],
    ut_path: Path,
) -> None:
    fig, axes = plt.subplots(len(HORISONTER), 1, figsize=(11, 10), sharex=True)
    for ax, h in zip(axes, HORISONTER):
        ax.plot(y_test.index, y_test.values, color="black", lw=1.2, label="Faktisk")

        s = sarima_prog[h]
        ax.plot(s.index, s["yhat"], color="tab:orange", lw=1.0, label="SARIMA")
        ax.fill_between(s.index, s["yhat_low"], s["yhat_high"], color="tab:orange", alpha=0.15)

        x = sarimax_prog[h]
        ax.plot(x.index, x["yhat"], color="tab:green", lw=1.0, label="SARIMAX (EUR/USD)")
        ax.fill_between(x.index, x["yhat_low"], x["yhat_high"], color="tab:green", alpha=0.15)

        ax.set_title(f"Horisont {h} uker - rullende opphav, 95% CI")
        ax.set_ylabel("NOK/kg")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(ut_path, dpi=120)
    plt.close(fig)
    print(f"  Lagret figur: {ut_path}")


def main() -> None:
    df = (
        pd.read_csv(DATA_DIR / "laks_ukentlig_features.csv", parse_dates=["uke_start"])
        .set_index("uke_start")
        .sort_index()
    )

    train = df.iloc[:-TEST_UKER]
    test = df.iloc[-TEST_UKER:]
    print(f"Train: {train.index.min().date()} - {train.index.max().date()} ({len(train)} uker)")
    print(f"Test:  {test.index.min().date()} - {test.index.max().date()} ({len(test)} uker)")

    y_train = train["eksport_pris_nok_kg"].asfreq("W-MON")
    y_test = test["eksport_pris_nok_kg"].asfreq("W-MON")

    # SARIMA (uten eksogene)
    sarima_prog, sarima_met, sarima_fit = rullerende_prognose(
        y_train, y_test, modell_navn="SARIMA",
    )

    # SARIMAX (med valuta)
    exog_train = train[EXOG_KOLS].asfreq("W-MON")
    exog_test = test[EXOG_KOLS].asfreq("W-MON")
    sarimax_prog, sarimax_met, sarimax_fit = rullerende_prognose(
        y_train, y_test,
        exog_train=exog_train, exog_test=exog_test,
        modell_navn="SARIMAX",
    )

    # Lagre metrikker
    metrikker = pd.concat([sarima_met, sarimax_met], ignore_index=True)
    metrikker.to_csv(UT_DIR / "sarima_metrikker.csv", index=False)
    print("\n--- Metrikker (rullende opphav) ---")
    print(metrikker.round(3).to_string(index=False))

    # Lagre prognoser med CI per modell og horisont
    for h in HORISONTER:
        sarima_prog[h].to_csv(UT_DIR / f"sarima_prognose_h{h}.csv")
        sarimax_prog[h].to_csv(UT_DIR / f"sarimax_prognose_h{h}.csv")

    # Plot prognoser
    plot_prognoser(y_test, sarima_prog, sarimax_prog, UT_DIR / "sarima_prognoser.png")

    # CI-kalibrering
    dekning = pd.concat([
        ci_dekning(sarima_prog, y_test, "SARIMA"),
        ci_dekning(sarimax_prog, y_test, "SARIMAX"),
    ], ignore_index=True)
    dekning.to_csv(UT_DIR / "sarima_ci_dekning.csv", index=False)
    print("\n--- CI-dekning (mal: 0.95) ---")
    print(dekning.round(3).to_string(index=False))

    # Residualdiagnostikk pa treningsfit
    diag_rader = [
        residualdiagnostikk(sarima_fit, "SARIMA", UT_DIR),
        residualdiagnostikk(sarimax_fit, "SARIMAX", UT_DIR),
    ]
    diag = pd.DataFrame(diag_rader)
    diag.to_csv(UT_DIR / "sarima_residualdiagnostikk.csv", index=False)
    print("\n--- Residualdiagnostikk (Ljung-Box og Jarque-Bera) ---")
    print(diag.round(4).to_string(index=False))

    print(f"\nLagret resultater til {UT_DIR}")


if __name__ == "__main__":
    main()
