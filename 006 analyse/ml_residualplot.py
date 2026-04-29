"""ml_residualplot.py – Residualanalyse for Ensemble-modellen (Spor B).

Leser ml_ensemble_prediksjoner.csv (generert av ml_ensemble.py) og produserer:
  1. residualplot_tid.png     – residualar over tid per horisont
  2. residualplot_scatter.png – faktisk vs. predikert scatter per horisont
  3. residualplot_hist.png    – histogram av residualar per horisont
  4. ml_residualar.csv        – rådata (uke_start, horisont, faktisk, pred, residual)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

UT_DIR = Path(__file__).parent / "resultater"
UT_DIR.mkdir(exist_ok=True)

HORISONTER = [4, 8, 12]
FARGAR = {4: "#FF5722", 8: "#2196F3", 12: "#4CAF50"}


def main() -> None:
    pred_path = UT_DIR / "ml_ensemble_prediksjoner.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            "ml_ensemble_prediksjoner.csv ikkje funne – køyr ml_ensemble.py først."
        )

    df = pd.read_csv(pred_path, parse_dates=["uke_start"])
    df["residual"] = df["faktisk"] - df["ensemble_pred"]
    df.to_csv(UT_DIR / "ml_residualar.csv", index=False)

    print(f"Lasta {len(df)} rader frå {pred_path.name}")
    for h in HORISONTER:
        sub = df[df["horisont"] == h].dropna(subset=["residual"])
        mae = sub["residual"].abs().mean()
        bias = sub["residual"].mean()
        std = sub["residual"].std()
        print(f"  h={h:2d}: n={len(sub):3d} | MAE={mae:.2f} | bias={bias:+.2f} | std={std:.2f}")

    # ------------------------------------------------------------------
    # 1. Residualar over tid
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(len(HORISONTER), 1, figsize=(13, 10), sharex=False)
    for ax, h in zip(axes, HORISONTER):
        sub = df[df["horisont"] == h].dropna(subset=["residual"]).sort_values("uke_start")
        farge = FARGAR[h]
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between(sub["uke_start"], sub["residual"], 0, alpha=0.25, color=farge)
        ax.plot(sub["uke_start"], sub["residual"], color=farge, linewidth=1.2)
        mae = sub["residual"].abs().mean()
        bias = sub["residual"].mean()
        ax.set_title(f"h={h} veker | Ensemble | MAE={mae:.2f} NOK/kg | bias={bias:+.2f}", fontsize=11)
        ax.set_ylabel("Residual (NOK/kg)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Dato")
    fig.suptitle("Residualar over tid – Ensemble (XGBoost + LightGBM)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(UT_DIR / "residualplot_tid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Lagra residualplot_tid.png")

    # ------------------------------------------------------------------
    # 2. Faktisk vs. predikert scatter
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(HORISONTER), figsize=(14, 5))
    for ax, h in zip(axes, HORISONTER):
        sub = df[df["horisont"] == h].dropna(subset=["ensemble_pred"])
        farge = FARGAR[h]
        ax.scatter(sub["faktisk"], sub["ensemble_pred"], alpha=0.5, color=farge, s=20)
        mn = min(sub["faktisk"].min(), sub["ensemble_pred"].min()) - 2
        mx = max(sub["faktisk"].max(), sub["ensemble_pred"].max()) + 2
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.9, label="Perfekt prediksjon")
        r2 = np.corrcoef(sub["faktisk"], sub["ensemble_pred"])[0, 1] ** 2
        mae = (sub["faktisk"] - sub["ensemble_pred"]).abs().mean()
        ax.set_title(f"h={h} veker\nMAE={mae:.2f} | R²={r2:.3f}", fontsize=11)
        ax.set_xlabel("Faktisk pris (NOK/kg)")
        ax.set_ylabel("Predikert pris (NOK/kg)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Faktisk vs. predikert – Ensemble", fontsize=13)
    fig.tight_layout()
    fig.savefig(UT_DIR / "residualplot_scatter.png", dpi=120)
    plt.close(fig)
    print("  Lagra residualplot_scatter.png")

    # ------------------------------------------------------------------
    # 3. Histogram over residualar
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(HORISONTER), figsize=(14, 4))
    for ax, h in zip(axes, HORISONTER):
        sub = df[df["horisont"] == h].dropna(subset=["residual"])
        farge = FARGAR[h]
        ax.hist(sub["residual"], bins=20, color=farge, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.axvline(sub["residual"].mean(), color="darkred", linewidth=1.2,
                   linestyle="-", label=f"Bias={sub['residual'].mean():+.2f}")
        ax.set_title(f"h={h} veker", fontsize=11)
        ax.set_xlabel("Residual (NOK/kg)")
        ax.set_ylabel("Frekvens")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Fordeling av residualar – Ensemble", fontsize=13)
    fig.tight_layout()
    fig.savefig(UT_DIR / "residualplot_hist.png", dpi=120)
    plt.close(fig)
    print("  Lagra residualplot_hist.png")

    print(f"\nAlle filer lagra til: {UT_DIR}")


if __name__ == "__main__":
    main()
