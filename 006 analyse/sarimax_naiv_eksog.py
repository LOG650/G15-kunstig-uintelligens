"""sarimax_naiv_eksog.py – SARIMAX med naiv (random walk) valutakursprognose.

I den opprinnelige SARIMAX-evalueringen ble de *faktiske* (realiserte) valutakursene
for prognoseperioden brukt som eksogene inndata (oracle-antagelse). Dette gir en
optimistisk øvre grense som ikke er direkte sammenlignbar med de øvrige modellene.

Dette skriptet kjører identisk walk-forward-evaluering, men bruker en **naiv
random-walk-prognose** for eksogene variabler: ved hvert steg t benyttes den
siste kjente kursverdien som prognose for alle fremtidige steg. Dette er den
standardprognosen som brukes i valuta-prognoselitteraturen (Meese & Rogoff, 1983)
og utgjør en rettferdig nedre grense for hva SARIMAX kan forvente i operativt bruk.

Resultater lagres til:
  resultater/sarimax_naiv_metrikker.csv
  resultater/sarimax_naiv_prognose_h{4,8,12}.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "004 data" / "Analyseklart datasett"
UT_DIR = Path(__file__).parent / "resultater"
UT_DIR.mkdir(exist_ok=True)

HORISONTER = [4, 8, 12]
TEST_UKER = 104
EXOG_KOLS = ["eur_nok_snitt", "usd_nok_snitt"]
ALPHA = 0.05

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


def rullerende_prognose_naiv_eksog(
    y_train: pd.Series,
    y_test: pd.Series,
    exog_train: pd.DataFrame,
    exog_test: pd.DataFrame,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Walk-forward SARIMAX med naiv random-walk-prognose for eksogene variabler.

    Ved hvert steg t brukes den siste *kjente* verdien av eksogene variabler
    (dvs. verdien ved t) som prognose for alle fremtidige steg t+1 ... t+h_max.
    Dette er den enkleste og mest brukte proxy for valutakursprognose i praksis.
    """
    modell_navn = "SARIMAX_naiv"
    print(f"\n[{modell_navn}] Tilpasser på {len(y_train)} treningspunkter ...")
    init = SARIMAX(
        y_train,
        exog=exog_train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    print(f"[{modell_navn}] AIC={init.aic:.1f}")

    h_max = max(HORISONTER)
    prognoser: dict[int, dict] = {h: {} for h in HORISONTER}
    n_test = len(y_test)
    current = init

    # Kombiner trenings- og testeksog for å ha tilgang til «siste kjente» verdi
    exog_full = pd.concat([exog_train, exog_test])

    print(f"[{modell_navn}] Walk-forward over {n_test} testuker ...")
    for step in range(n_test):
        # Indeks i det fulle eksog-settet som tilsvarer «nå»
        full_idx = len(exog_train) + step

        # Siste kjente eksog-verdi (random-walk-prognose: hold konstant)
        last_known = exog_full.iloc[full_idx].values.reshape(1, -1)
        exog_future = pd.DataFrame(
            np.repeat(last_known, h_max, axis=0),
            columns=EXOG_KOLS,
        )

        fc = current.get_forecast(steps=h_max, exog=exog_future)
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

        nytt_y = y_test.iloc[step : step + 1]
        nytt_exog = exog_test.iloc[step : step + 1]
        current = current.append(nytt_y, exog=nytt_exog, refit=False)

        if (step + 1) % 26 == 0 or step == n_test - 1:
            print(f"[{modell_navn}]   {step + 1}/{n_test} uker behandlet")

    prognose_dfs: dict[int, pd.DataFrame] = {}
    metrikker_rader = []
    for h in HORISONTER:
        rows = sorted(prognoser[h].items())
        idx = [d for d, _ in rows]
        vals = np.array([v for _, v in rows])
        df = pd.DataFrame(
            vals,
            index=pd.DatetimeIndex(idx, name="uke_start"),
            columns=["yhat", "yhat_low", "yhat_high"],
        )
        prognose_dfs[h] = df
        metrikker_rader.append(
            evaluer(y_test.loc[df.index], df["yhat"], modell_navn, h)
        )

    return prognose_dfs, pd.DataFrame(metrikker_rader)


def main() -> None:
    df = (
        pd.read_csv(DATA_DIR / "laks_ukentlig_features.csv", parse_dates=["uke_start"])
        .set_index("uke_start")
        .sort_index()
    )
    train = df.iloc[:-TEST_UKER]
    test = df.iloc[-TEST_UKER:]

    y_train = train["eksport_pris_nok_kg"].asfreq("W-MON")
    y_test = test["eksport_pris_nok_kg"].asfreq("W-MON")
    exog_train = train[EXOG_KOLS].asfreq("W-MON")
    exog_test = test[EXOG_KOLS].asfreq("W-MON")

    prognose_dfs, metrikker = rullerende_prognose_naiv_eksog(
        y_train, y_test, exog_train, exog_test
    )

    metrikker.to_csv(UT_DIR / "sarimax_naiv_metrikker.csv", index=False)
    for h in HORISONTER:
        prognose_dfs[h].to_csv(UT_DIR / f"sarimax_naiv_prognose_h{h}.csv")

    print("\n--- SARIMAX_naiv vs. SARIMAX_oracle ---")
    oracle = pd.read_csv(UT_DIR / "sarima_metrikker.csv")
    oracle_sarimax = oracle[oracle["modell"] == "SARIMAX"].set_index("horisont")

    for _, row in metrikker.iterrows():
        h = int(row["horisont"])
        oracle_mae = oracle_sarimax.loc[h, "MAE"]
        naiv_mae = row["MAE"]
        diff = naiv_mae - oracle_mae
        print(
            f"  h={h}: Oracle MAE={oracle_mae:.2f} | Naiv MAE={naiv_mae:.2f} "
            f"| Diff={diff:+.2f} NOK/kg ({diff/oracle_mae*100:+.1f}%)"
        )

    print(f"\nLagret til {UT_DIR}")


if __name__ == "__main__":
    main()
