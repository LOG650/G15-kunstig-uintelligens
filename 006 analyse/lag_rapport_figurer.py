"""Spor D – Rapport-kvalitets-figurer.

Produserer tre rapport-klare figurer til rapporten:
  1. rapport_modellsammenligning.png/pdf  – MAE-sammenlikning alle modeller
  2. rapport_beste_prognose.png/pdf       – Beste prognose per horisont vs. faktisk pris
  3. rapport_ensemble_bias.png/pdf        – Ensemble bias-analyse

Leser fra: 006 analyse/resultater/
Skriver til: 006 analyse/resultater/rapport_*.png og rapport_*.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

UT_DIR = Path(__file__).parent / "resultater"
DATA_DIR = Path(__file__).parent.parent / "004 data" / "Analyseklart datasett"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

HORISONTER = [4, 8, 12]


# ---------------------------------------------------------------------------
# Figur 1 – Modell-sammenlikning (grouped bar chart)
# ---------------------------------------------------------------------------

def fig_modellsammenligning() -> None:
    modeller = [
        ("Naiv", 8.51, 13.04, 16.35),
        ("SARIMA", 8.27, 11.07, 13.15),
        ("SARIMAX", 8.33, 11.07, 12.93),
        ("XGB tunet", 10.37, 11.98, 15.47),
        ("LGBM tunet", 10.45, 10.90, 13.06),
        ("XGB+ES", 8.71, 10.88, 15.31),
        ("LGBM+ES", 8.85, 11.53, 13.24),
        ("Ensemble", 8.33, 10.85, 13.56),
    ]

    labels = [m[0] for m in modeller]
    mae_h4 = [m[1] for m in modeller]
    mae_h8 = [m[2] for m in modeller]
    mae_h12 = [m[3] for m in modeller]

    x = np.arange(len(labels))
    w = 0.25
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    b1 = ax.bar(x - w, mae_h4,  w, label="h = 4 uker",  color=colors[0], alpha=0.85)
    b2 = ax.bar(x,      mae_h8,  w, label="h = 8 uker",  color=colors[1], alpha=0.85)
    b3 = ax.bar(x + w, mae_h12, w, label="h = 12 uker", color=colors[2], alpha=0.85)

    # Merk de beste per horisont
    for i, label in enumerate(labels):
        if label == "SARIMA":
            ax.text(x[i] - w, mae_h4[i] + 0.1, "★", ha="center", fontsize=9, color=colors[0])
        if label == "Ensemble":
            ax.text(x[i], mae_h8[i] + 0.1, "★", ha="center", fontsize=9, color=colors[1])
        if label == "SARIMAX":
            ax.text(x[i] + w, mae_h12[i] + 0.1, "★", ha="center", fontsize=9, color=colors[2])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("MAE (NOK/kg)")
    ax.set_title("Prognoseytelse per modell og horisont   ★ = best per horisont")
    ax.legend(loc="upper right")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_ylim(0, 19)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(UT_DIR / f"rapport_modellsammenligning.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Lagret: rapport_modellsammenligning.png/.pdf")


# ---------------------------------------------------------------------------
# Figur 2 – Beste prognose per horisont vs. faktisk pris
# ---------------------------------------------------------------------------

def fig_beste_prognose() -> None:
    df = (
        pd.read_csv(DATA_DIR / "laks_ukentlig_features.csv", parse_dates=["uke_start"])
        .set_index("uke_start")
        .sort_index()
    )
    TEST_UKER = 104
    y_test = df["eksport_pris_nok_kg"].iloc[-TEST_UKER:]

    # Beste modeller: SARIMA h=4, Ensemble h=8, SARIMAX h=12
    sarima_h4 = pd.read_csv(UT_DIR / "sarima_prognose_h4.csv", parse_dates=["uke_start"]).set_index("uke_start")["yhat"]
    sarima_h4.index = pd.DatetimeIndex(sarima_h4.index)

    sarimax_h12 = pd.read_csv(UT_DIR / "sarimax_prognose_h12.csv", parse_dates=["uke_start"]).set_index("uke_start")["yhat"]
    sarimax_h12.index = pd.DatetimeIndex(sarimax_h12.index)

    ens_df = pd.read_csv(UT_DIR / "ml_ensemble_prediksjoner.csv", parse_dates=["uke_start"])
    ens_df = ens_df[(ens_df["horisont"] == 8) & ens_df["ensemble_pred"].notna()].set_index("uke_start")
    ensemble_h8 = ens_df["ensemble_pred"]
    ensemble_h8.index = pd.DatetimeIndex(ensemble_h8.index)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)

    pairs = [
        (axes[0], sarima_h4,  "h = 4 uker – SARIMA", "#2196F3"),
        (axes[1], ensemble_h8, "h = 8 uker – Ensemble (XGB+LGBM)", "#FF9800"),
        (axes[2], sarimax_h12, "h = 12 uker – SARIMAX (EUR/USD)", "#4CAF50"),
    ]

    for ax, yhat, tittel, clr in pairs:
        idx = yhat.index.intersection(y_test.index)
        ax.plot(y_test.loc[idx], color="black", lw=1.2, label="Faktisk pris", zorder=3)
        ax.plot(yhat.loc[idx], color=clr, lw=1.2, linestyle="--", label="Prognose", zorder=2, alpha=0.9)
        mae = np.mean(np.abs(y_test.loc[idx] - yhat.loc[idx]))
        ax.set_title(f"{tittel}   (MAE = {mae:.2f} NOK/kg)")
        ax.set_ylabel("NOK/kg")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")

    fig.suptitle("Beste prognose per horisont vs. faktisk eksportpris (testperiode 2022–2024)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(UT_DIR / f"rapport_beste_prognose.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Lagret: rapport_beste_prognose.png/.pdf")


# ---------------------------------------------------------------------------
# Figur 3 – Ensemble-bias analyse
# ---------------------------------------------------------------------------

def fig_ensemble_bias() -> None:
    try:
        residualar = pd.read_csv(UT_DIR / "ml_residualar.csv", parse_dates=["uke_start"])
    except FileNotFoundError:
        print("ml_residualar.csv ikke funnet – hopper over figur 3.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    colors_h = {4: "#2196F3", 8: "#FF9800", 12: "#4CAF50"}

    for ax, h in zip(axes, HORISONTER):
        sub = residualar[(residualar["horisont"] == h) & residualar["residual"].notna()].copy()
        if sub.empty:
            continue

        bias = sub["residual"].mean()
        std  = sub["residual"].std()

        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.axhline(bias, color=colors_h[h], lw=1.5, linestyle="-", label=f"Bias = {bias:+.2f} NOK/kg")
        ax.scatter(sub["faktisk"], sub["residual"], s=15, alpha=0.4, color=colors_h[h], edgecolors="none")
        ax.set_xlabel("Faktisk pris (NOK/kg)")
        ax.set_ylabel("Residual (faktisk – prognose)" if h == 4 else "")
        ax.set_title(f"h = {h} uker")
        ax.legend()
        ax.grid(alpha=0.2)

        ax.text(0.02, 0.97,
                f"n = {len(sub)}\nbias = {bias:+.2f}\nstd = {std:.2f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.suptitle("Ensemble residualer: faktisk pris vs. prediksjonsfeil (testperiode)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(UT_DIR / f"rapport_ensemble_bias.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Lagret: rapport_ensemble_bias.png/.pdf")


# ---------------------------------------------------------------------------
# Figur 4 – CI-kalibrering (bonus)
# ---------------------------------------------------------------------------

def fig_ci_kalibrering() -> None:
    try:
        kal = pd.read_csv(UT_DIR / "usikkerhet_kalibrering.csv")
    except FileNotFoundError:
        return

    metoder_order = [
        ("SARIMA_gauss",      "SARIMA Gauss",         "#1565C0"),
        ("SARIMAX_gauss",     "SARIMAX Gauss",         "#1E88E5"),
        ("SARIMA_bootstrap",  "SARIMA Bootstrap",      "#EF6C00"),
        ("SARIMAX_bootstrap", "SARIMAX Bootstrap",     "#FB8C00"),
        ("LightGBM_quantile", "LGBM Kvantilregr.",    "#2E7D32"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)

    for ax, h in zip(axes, HORISONTER):
        sub = kal[kal["horisont"] == h]
        y_vals = []
        x_labels = []
        colors = []
        for modell_id, modell_navn, clr in metoder_order:
            row = sub[sub["modell"] == modell_id]
            if not row.empty:
                y_vals.append(row["dekning"].values[0] * 100)
                x_labels.append(modell_navn)
                colors.append(clr)

        bars = ax.bar(range(len(x_labels)), y_vals, color=colors, width=0.6, alpha=0.85)
        ax.axhline(95, color="red", lw=1.2, linestyle="--", label="Nominelt mål 95 %")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"h = {h} uker")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)

        for bar, val in zip(bars, y_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Empirisk dekning (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("CI-kalibrering: empirisk dekning av nominell 95 %-intervall", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(UT_DIR / f"rapport_ci_kalibrering.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Lagret: rapport_ci_kalibrering.png/.pdf")


# ---------------------------------------------------------------------------
# Hovedprogram
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.dates

    print("=== Rapport-figurer ===")

    fig_modellsammenligning()
    fig_beste_prognose()
    fig_ensemble_bias()
    fig_ci_kalibrering()

    print("\nFerdig. Alle rapport-figurer i:", UT_DIR)
