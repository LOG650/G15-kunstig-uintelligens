"""diebold_mariano.py – Diebold-Mariano-test for lik prediksjonsevne.

Tester H0: modell A og modell B har lik forventet prediksjonsnøyaktighet.
Bruker Harvey, Leybourne & Newbold (1997) small-sample-korreksjon og
Newey-West HAC-varians for å håndtere autokorrelasjon i tapsdifferansen
(særlig relevant for multi-steg-prognoser).

Referanse:
  Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of
  prediction mean squared errors. International Journal of Forecasting,
  13(2), 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4

  Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy.
  Journal of Business & Economic Statistics, 13(3), 253–263.

Sammenligninger kjøres per horisont:
  h=4:  SARIMA  vs. Ensemble
  h=8:  Ensemble vs. SARIMA
  h=12: SARIMAX_naiv vs. LightGBM+ES

Resultater lagres til resultater/diebold_mariano.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

UT_DIR = Path(__file__).parent / "resultater"
DATA_DIR = Path(__file__).parent.parent / "004 data" / "Analyseklart datasett"


# ---------------------------------------------------------------------------
# Hjelpe-funksjoner
# ---------------------------------------------------------------------------

def newey_west_variance(d: np.ndarray, h: int) -> float:
    """Newey-West HAC-varians av tapsdifferansen d med båndbredde h-1.

    For h-steg-frem-prognoser er den teoretiske autokorrelasjonslengden h-1
    (Diebold & Mariano, 1995). Bruker Bartlett-kjerne.
    """
    n = len(d)
    gamma0 = np.var(d, ddof=1)
    lag = h - 1
    if lag == 0:
        return gamma0
    acovs = []
    for k in range(1, lag + 1):
        cov_k = np.mean((d[k:] - d.mean()) * (d[:-k] - d.mean()))
        weight = 1.0 - k / (lag + 1)  # Bartlett-vekt
        acovs.append(weight * cov_k)
    return gamma0 + 2 * sum(acovs)


def dm_test(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    h: int,
    loss: str = "mae",
) -> dict:
    """Diebold-Mariano-test med Harvey-Leybourne-Newbold small-sample-korreksjon.

    H0: E[L(e1)] = E[L(e2)]  (lik prediksjonsevne)
    H1: E[L(e1)] ≠ E[L(e2)]  (to-sidig, MAPE-skala)

    Positiv DM-statistikk betyr at modell 1 har høyere tap enn modell 2
    (dvs. modell 2 er bedre). Negativ betyr at modell 1 er bedre.

    Returnerer dict med DM-statistikk, p-verdi og tolkning.
    """
    mask = ~(np.isnan(actual) | np.isnan(pred1) | np.isnan(pred2))
    a, p1, p2 = actual[mask], pred1[mask], pred2[mask]
    n = len(a)

    e1 = a - p1
    e2 = a - p2

    if loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:  # mse
        d = e1 ** 2 - e2 ** 2

    d_bar = d.mean()
    nw_var = newey_west_variance(d, h)

    # Harvey-Leybourne-Newbold small-sample-korreksjon
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat = d_bar / np.sqrt(nw_var / n)
    dm_hln = dm_stat * hln_factor

    # t-fordeling med n-1 frihetsgrader (HLN-korreksjon)
    p_value = 2 * stats.t.sf(abs(dm_hln), df=n - 1)

    return {
        "n": n,
        "d_bar": d_bar,
        "dm_stat": dm_stat,
        "dm_hln": dm_hln,
        "p_value": p_value,
        "signifikant_5pct": p_value < 0.05,
        "signifikant_10pct": p_value < 0.10,
    }


# ---------------------------------------------------------------------------
# Last data og prognoser
# ---------------------------------------------------------------------------

def last_aktuelle_og_prognose(
    h: int,
    modell: str,
) -> pd.Series:
    """Henter prognoseserie for angitt modell og horisont."""
    if modell in ("SARIMA", "SARIMAX", "SARIMAX_naiv"):
        prefix = modell.lower()
        path = UT_DIR / f"{prefix}_prognose_h{h}.csv"
        df = pd.read_csv(path, parse_dates=["uke_start"]).set_index("uke_start")
        return df["yhat"]
    elif modell in ("XGBoost+ES", "LightGBM+ES", "Ensemble"):
        pred_df = pd.read_csv(
            UT_DIR / "ml_ensemble_prediksjoner.csv", parse_dates=["uke_start"]
        )
        sub = pred_df[pred_df["horisont"] == h].set_index("uke_start")
        col_map = {"XGBoost+ES": "xgb_pred", "LightGBM+ES": "lgbm_pred", "Ensemble": "ensemble_pred"}
        return sub[col_map[modell]]
    else:
        raise ValueError(f"Ukjent modell: {modell}")


def last_faktisk(h: int) -> pd.Series:
    """Henter faktisk prisserie fra ML-prediksjonsfilen."""
    pred_df = pd.read_csv(
        UT_DIR / "ml_ensemble_prediksjoner.csv", parse_dates=["uke_start"]
    )
    sub = pred_df[pred_df["horisont"] == h].set_index("uke_start")
    return sub["faktisk"]


# ---------------------------------------------------------------------------
# Hovudprogram
# ---------------------------------------------------------------------------

def main() -> None:
    sammenligninger = [
        # (horisont, modell1 (referanse), modell2 (utfordrer), loss)
        # h=4: Er SARIMA og Ensemble statistisk like? (SARIMA er marginalt bedre)
        (4, "SARIMA",       "Ensemble",      "mae"),
        # h=8: Er Ensemble bedre enn SARIMA?
        (8, "SARIMA",       "Ensemble",      "mae"),
        # h=8: Er Ensemble bedre enn LightGBM+ES?
        (8, "LightGBM+ES",  "Ensemble",      "mae"),
        # h=12: Er SARIMAX_naiv og LightGBM+ES statistisk like?
        (12, "LightGBM+ES", "SARIMAX_naiv",  "mae"),
        # h=12: Oracle SARIMAX vs. naiv SARIMAX
        (12, "SARIMAX",     "SARIMAX_naiv",  "mae"),
    ]

    resultater = []
    print("\n" + "=" * 70)
    print("DIEBOLD-MARIANO TESTER (Harvey-Leybourne-Newbold small-sample)")
    print("=" * 70)
    print(f"{'h':>4} | {'Modell 1':>14} | {'Modell 2':>14} | {'d_bar':>8} | {'DM_HLN':>8} | {'p-verdi':>8} | {'Sig?':>5}")
    print("-" * 70)

    for h, m1, m2, loss in sammenligninger:
        faktisk = last_faktisk(h)
        p1 = last_aktuelle_og_prognose(h, m1)
        p2 = last_aktuelle_og_prognose(h, m2)

        # Aligner på felles indeks
        felles = faktisk.index.intersection(p1.index).intersection(p2.index)
        a = faktisk.loc[felles].values
        f1 = p1.loc[felles].values
        f2 = p2.loc[felles].values

        res = dm_test(a, f1, f2, h, loss)
        sig = "**" if res["signifikant_5pct"] else ("*" if res["signifikant_10pct"] else "n.s.")

        print(
            f"  {h:>2} | {m1:>14} | {m2:>14} | {res['d_bar']:>+8.3f} | "
            f"{res['dm_hln']:>+8.3f} | {res['p_value']:>8.4f} | {sig:>5}"
        )

        resultater.append({
            "horisont": h,
            "modell_1": m1,
            "modell_2": m2,
            "tap": loss,
            "n": res["n"],
            "d_bar_nok_kg": round(res["d_bar"], 4),
            "dm_stat": round(res["dm_stat"], 4),
            "dm_hln": round(res["dm_hln"], 4),
            "p_verdi": round(res["p_value"], 4),
            "signifikant_5pct": res["signifikant_5pct"],
            "signifikant_10pct": res["signifikant_10pct"],
        })

    print()
    print("* p < 0.10   ** p < 0.05   n.s. = ikke signifikant")
    print("Positiv d_bar: modell 1 har høyere tap (modell 2 er bedre)")

    res_df = pd.DataFrame(resultater)
    res_df.to_csv(UT_DIR / "diebold_mariano.csv", index=False)
    print(f"\nLagret til {UT_DIR / 'diebold_mariano.csv'}")


if __name__ == "__main__":
    main()
