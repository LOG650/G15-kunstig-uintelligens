"""Spor E – Auto-ARIMA verifikasjon og refit-sensitivitetsanalyse.

Validerer SARIMA-ordensvalget fra Spor A med pmdarima.auto_arima, og undersøker
sensitiviteten til walk-forward-evalueringen mhp. refit-frekvens.

Leser fra:   004 data/Analyseklart datasett/laks_ukentlig_features.csv
             006 analyse/resultater/sarima_metrikker.csv  (Spor A-baseline)
Skriver til: 006 analyse/resultater/sarima_avansert_*.csv og *.png
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
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

# Spor A sin fastsatte orden – dette er baseline vi validerer
SPOR_A_ORDER = (1, 1, 1)
SPOR_A_SEASONAL = (1, 1, 1, 52)

# Refit-frekvenser som testes (antall uker mellom hver refit; inf = aldri)
# refit=1 utelatt: krever 104 fulle SARIMA-refits (~40-90 min) – ikke praktisk nytte
REFIT_FREKVENSER = [4, 12, 26, float("inf")]
REFIT_NAVN = {4: "4", 12: "12", 26: "26", float("inf"): "inf"}


# ---------------------------------------------------------------------------
# Hjelpefunksjoner (kopiert og utvidet fra sarima_eksperiment.py)
# ---------------------------------------------------------------------------

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


def walk_forward(
    y_train: pd.Series,
    y_test: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    modell_navn: str = "SARIMA",
    refit_frekvens: float = float("inf"),
) -> tuple[pd.DataFrame, float]:
    """Walk-forward forecast med valgbar refit-frekvens.

    refit_frekvens:
      1   = refit hvert steg  (sakte – estimerer params pa nytt 104 ganger)
      4   = refit hver 4. uke
      12  = refit hver 12. uke
      26  = refit hver 26. uke
      inf = aldri (Spor A sin hurtiglosning, refit=False alltid)

    Returnerer (metrikker_df, kjoretid_sekunder).
    """
    print(f"\n[{modell_navn}] Tilpasser pa {len(y_train)} treningspunkter ...")
    t_start = time.time()

    init = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    print(f"  Treningsfit: AIC={init.aic:.1f}, BIC={init.bic:.1f}")

    prognoser: dict[int, dict] = {h: {} for h in HORISONTER}
    h_max = max(HORISONTER)
    current = init
    n_test = len(y_test)

    for step in range(n_test):
        fc = current.get_forecast(steps=h_max)
        fc_mean = fc.predicted_mean.values

        for h in HORISONTER:
            target_step = step + h - 1
            if target_step < n_test:
                target_date = y_test.index[target_step]
                prognoser[h][target_date] = fc_mean[h - 1]

        nytt_y = y_test.iloc[step : step + 1]
        ska_refit = (refit_frekvens != float("inf")) and ((step + 1) % refit_frekvens == 0)
        current = current.append(nytt_y, refit=ska_refit)

        if (step + 1) % 26 == 0 or step == n_test - 1:
            print(f"  {step + 1}/{n_test} uker behandlet")

    elapsed = time.time() - t_start

    metrikker = []
    for h in HORISONTER:
        rows = sorted(prognoser[h].items())
        if not rows:
            continue
        idx = [d for d, _ in rows]
        yhat = pd.Series([v for _, v in rows], index=pd.DatetimeIndex(idx, name="uke_start"))
        metrikker.append(evaluer(y_test.loc[yhat.index], yhat, modell_navn, h))

    return pd.DataFrame(metrikker), elapsed


# ---------------------------------------------------------------------------
# Steg 1 – Auto-ARIMA
# ---------------------------------------------------------------------------

def run_autoarima(y_train: pd.Series):
    """Kjoer auto-ARIMA og returner (pmdarima_modell, kjoretid_sek).

    Returnerer (None, 0) hvis pmdarima ikke er installert.
    """
    try:
        import pmdarima as pm
    except ImportError:
        print("\n[auto-ARIMA] pmdarima ikke installert.")
        print("  Installer med: pip install pmdarima")
        print("  Hopper over steg 1 – bruker Spor A sin orden som fallback.")
        return None, 0

    print("\n[auto-ARIMA] Starter ordensøk (kan ta 10–30 min) ...")
    t_start = time.time()

    model = pm.auto_arima(
        y_train,
        seasonal=True,
        m=52,
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        max_d=2,
        max_D=1,
        stepwise=True,
        information_criterion="aic",
        error_action="ignore",
        suppress_warnings=True,
        trace=True,
    )

    elapsed = time.time() - t_start
    print(f"\n[auto-ARIMA] Ferdig pa {elapsed / 60:.1f} min")
    print(f"  Foreslatt orden:  ARIMA{model.order}  x  {model.seasonal_order}")
    print(f"  AIC={model.aic():.1f}, BIC={model.bic():.1f}")
    return model, elapsed


def bygg_autoarima_csv(
    auto_model,
    spor_a_fit,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Bygg sammenlikningstabellen for auto-ARIMA vs. Spor A."""

    rader = []

    # Spor A sin rad
    try:
        spor_a_baseline = pd.read_csv(UT_DIR / "sarima_metrikker.csv")
        spor_a_sarima = spor_a_baseline[spor_a_baseline["modell"] == "SARIMA"]
    except FileNotFoundError:
        spor_a_sarima = pd.DataFrame()

    rad_a: dict = {
        "modell": "SARIMA(1,1,1)(1,1,1,52)_SporA",
        "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "m": 52,
        "AIC": spor_a_fit.aic,
        "BIC": spor_a_fit.bic,
        "kilde": "manuell",
        "MAE_h4": np.nan, "MAE_h8": np.nan, "MAE_h12": np.nan,
        "MAPE_h4": np.nan, "MAPE_h8": np.nan, "MAPE_h12": np.nan,
    }
    for _, row in spor_a_sarima.iterrows():
        h = int(row["horisont"])
        rad_a[f"MAE_h{h}"] = row["MAE"]
        rad_a[f"MAPE_h{h}"] = row["MAPE"]
    rader.append(rad_a)

    if auto_model is None:
        return pd.DataFrame(rader)

    auto_order = auto_model.order
    auto_seasonal = auto_model.seasonal_order  # (P, D, Q, m)

    # Fit auto-ordren med statsmodels for AIC/BIC pa samme grunnlag som Spor A
    auto_sm = SARIMAX(
        y_train,
        order=auto_order,
        seasonal_order=auto_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    rad_auto: dict = {
        "modell": f"SARIMA{auto_order}{auto_seasonal}_autoARIMA",
        "p": auto_order[0], "d": auto_order[1], "q": auto_order[2],
        "P": auto_seasonal[0], "D": auto_seasonal[1], "Q": auto_seasonal[2], "m": auto_seasonal[3],
        "AIC": auto_sm.aic,
        "BIC": auto_sm.bic,
        "kilde": "auto_arima",
        "MAE_h4": np.nan, "MAE_h8": np.nan, "MAE_h12": np.nan,
        "MAPE_h4": np.nan, "MAPE_h8": np.nan, "MAPE_h12": np.nan,
    }

    ordre_lik = (auto_order == SPOR_A_ORDER) and (auto_seasonal[:3] == SPOR_A_SEASONAL[:3])
    if ordre_lik:
        print("\n[auto-ARIMA] Orden er identisk med Spor A – walk-forward ikke nodvendig.")
        for _, row in spor_a_sarima.iterrows():
            h = int(row["horisont"])
            rad_auto[f"MAE_h{h}"] = row["MAE"]
            rad_auto[f"MAPE_h{h}"] = row["MAPE"]
    else:
        print(f"\n[auto-ARIMA] Alternativ orden funnet – kjorer walk-forward ...")
        auto_met, _ = walk_forward(
            y_train, y_test,
            order=auto_order,
            seasonal_order=auto_seasonal,
            modell_navn=f"auto-ARIMA{auto_order}{auto_seasonal}",
        )
        for _, row in auto_met.iterrows():
            h = int(row["horisont"])
            rad_auto[f"MAE_h{h}"] = row["MAE"]
            rad_auto[f"MAPE_h{h}"] = row["MAPE"]

    rader.append(rad_auto)
    return pd.DataFrame(rader)


# ---------------------------------------------------------------------------
# Steg 2 – Refit-sensitivitetsanalyse
# ---------------------------------------------------------------------------

def kjor_refit_sensitivitet(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Kjor walk-forward for alle refit-frekvenser og returner resultattabell."""
    rader = []

    for rf in REFIT_FREKVENSER:
        rf_str = REFIT_NAVN[rf]
        modell_navn = f"SARIMA_refit{rf_str}"

        met, elapsed = walk_forward(
            y_train, y_test,
            order=SPOR_A_ORDER,
            seasonal_order=SPOR_A_SEASONAL,
            modell_navn=modell_navn,
            refit_frekvens=rf,
        )
        print(f"  Kjoretid: {elapsed:.0f} sek")

        for _, row in met.iterrows():
            rader.append({
                "refit_hver_uke": rf_str,
                "refit_frekvens_num": int(rf) if rf != float("inf") else 9999,
                "horisont": int(row["horisont"]),
                "n": int(row["n"]),
                "MAE": row["MAE"],
                "MAPE": row["MAPE"],
                "sek_kjoretid": round(elapsed, 1),
            })

    return pd.DataFrame(rader)


# ---------------------------------------------------------------------------
# Figurer
# ---------------------------------------------------------------------------

def plot_refit_sensitivitet(refit_df: pd.DataFrame) -> None:
    rekkefolge = ["4", "12", "26", "inf"]
    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, h in zip(axes, HORISONTER):
        subset = (
            refit_df[refit_df["horisont"] == h]
            .set_index("refit_hver_uke")
            .reindex(rekkefolge)
        )
        bars = ax.bar(rekkefolge, subset["MAE"], color=colors, width=0.6)
        baseline = subset.loc["inf", "MAE"]
        ax.axhline(baseline, color="black", linestyle="--", lw=1.0, label="refit=inf (Spor A)")

        for bar, val in zip(bars, subset["MAE"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_title(f"h={h} uker")
        ax.set_xlabel("Refit hver N uke")
        ax.set_ylabel("MAE (NOK/kg)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Refit-sensitivitet – SARIMA(1,1,1)(1,1,1,52)", fontsize=12)
    plt.tight_layout()
    fig.savefig(UT_DIR / "sarima_avansert_refit_sensitivitet.png", dpi=120)
    plt.close(fig)
    print("  Lagret: sarima_avansert_refit_sensitivitet.png")


def plot_aic_bic(autoarima_df: pd.DataFrame) -> None:
    if len(autoarima_df) < 2:
        return
    etiketter = [
        r.replace("_SporA", "\n(Spor A)").replace("_autoARIMA", "\n(auto-ARIMA)")
        for r in autoarima_df["modell"]
    ]
    x = np.arange(len(autoarima_df))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, autoarima_df["AIC"], width=0.35, label="AIC", color="tab:blue")
    ax.bar(x + 0.2, autoarima_df["BIC"], width=0.35, label="BIC", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(etiketter, fontsize=9)
    ax.set_ylabel("Informasjonskriterium")
    ax.set_title("AIC/BIC: Spor A vs. auto-ARIMA")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(UT_DIR / "sarima_avansert_aic_bic.png", dpi=120)
    plt.close(fig)
    print("  Lagret: sarima_avansert_aic_bic.png")


# ---------------------------------------------------------------------------
# Hovedprogram
# ---------------------------------------------------------------------------

def main() -> None:
    df = (
        pd.read_csv(DATA_DIR / "laks_ukentlig_features.csv", parse_dates=["uke_start"])
        .set_index("uke_start")
        .sort_index()
    )

    train = df.iloc[:-TEST_UKER]
    test = df.iloc[-TEST_UKER:]
    print(f"Train: {train.index.min().date()} – {train.index.max().date()} ({len(train)} uker)")
    print(f"Test:  {test.index.min().date()} – {test.index.max().date()} ({len(test)} uker)")

    y_train = train["eksport_pris_nok_kg"].asfreq("W-MON")
    y_test = test["eksport_pris_nok_kg"].asfreq("W-MON")

    # Fit Spor A sin orden for AIC/BIC-sammenligning
    print("\n[Spor A baseline] Tilpasser SARIMA(1,1,1)(1,1,1,52) ...")
    spor_a_fit = SARIMAX(
        y_train,
        order=SPOR_A_ORDER,
        seasonal_order=SPOR_A_SEASONAL,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    print(f"  AIC={spor_a_fit.aic:.1f}, BIC={spor_a_fit.bic:.1f}")

    # ------------------------------------------------------------------
    # STEG 1: Auto-ARIMA
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEG 1: Auto-ARIMA ordensverifikasjon")
    print("=" * 60)

    auto_model, _ = run_autoarima(y_train)

    autoarima_df = bygg_autoarima_csv(auto_model, spor_a_fit, y_train, y_test)
    autoarima_df.to_csv(UT_DIR / "sarima_avansert_autoarima.csv", index=False)
    print("\n--- Auto-ARIMA sammenligning ---")
    print(autoarima_df.round(3).to_string(index=False))

    # ------------------------------------------------------------------
    # STEG 2: Refit-sensitivitetsanalyse
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEG 2: Refit-sensitivitetsanalyse")
    print("=" * 60)

    refit_df = kjor_refit_sensitivitet(y_train, y_test)
    refit_df.to_csv(UT_DIR / "sarima_avansert_refit_sensitivitet.csv", index=False)

    print("\n--- MAE per refit-frekvens og horisont ---")
    pivot = refit_df.pivot_table(values="MAE", index="refit_hver_uke", columns="horisont")
    pivot = pivot.reindex(["1", "4", "12", "26", "inf"])
    print(pivot.round(3).to_string())

    print("\n--- Kjoretid per refit-frekvens ---")
    tid = refit_df.drop_duplicates("refit_hver_uke").set_index("refit_hver_uke")["sek_kjoretid"]
    print(tid.reindex(["1", "4", "12", "26", "inf"]).to_string())

    # ------------------------------------------------------------------
    # Figurer
    # ------------------------------------------------------------------
    print("\nLager figurer ...")
    plot_refit_sensitivitet(refit_df)
    plot_aic_bic(autoarima_df)

    # ------------------------------------------------------------------
    # Oppsummering
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OPPSUMMERING – Spor E")
    print("=" * 60)

    if auto_model is not None:
        auto_order = auto_model.order
        auto_seasonal = auto_model.seasonal_order
        ordre_lik = (auto_order == SPOR_A_ORDER) and (auto_seasonal[:3] == SPOR_A_SEASONAL[:3])
        if ordre_lik:
            print(f"Auto-ARIMA BEKREFTER Spor A sin orden: {SPOR_A_ORDER} x {SPOR_A_SEASONAL}")
        else:
            print(f"Auto-ARIMA foreslar ALTERNATIV orden: {auto_order} x {auto_seasonal}")
            print(f"  Spor A brukte:                        {SPOR_A_ORDER} x {SPOR_A_SEASONAL}")
            print("  Se sarima_avansert_autoarima.csv for MAE-sammenligning.")
    else:
        print("pmdarima ikke tilgjengelig – auto-ARIMA-steg hoppet over.")

    baseline_mae = refit_df[refit_df["refit_hver_uke"] == "inf"].set_index("horisont")["MAE"]
    print("\nRefit-sensitivitet (delta MAE vs. refit=inf, som er Spor A sin tilnærming):")
    for rf in ["4", "12", "26"]:
        rad = refit_df[refit_df["refit_hver_uke"] == rf].set_index("horisont")["MAE"]
        diffs = [f"h={h}: {rad[h] - baseline_mae[h]:+.3f}" for h in HORISONTER if h in rad.index]
        print(f"  refit={rf}: {', '.join(diffs)}")

    print(f"\nFerdig. Resultater i {UT_DIR}")


if __name__ == "__main__":
    main()
