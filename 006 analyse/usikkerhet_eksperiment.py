"""Spor G - Empirisk kalibrerte konfidensintervall.

Steg 1: Bootstrap-CI for SARIMA og SARIMAX.
         In-sample 1-steg-feil (fittedvalues vs. faktisk, skip 52 init-obs)
         samplas med erstatning og adderas til eksisterande punktprognose.
         Empirisk 2.5/97.5-persentil gjev kalibrert CI.
Steg 2: Kvantilregresjon med LightGBM (q=0.025, 0.5, 0.975 per horisont).
         Same features og split som ml_ensemble.py.
Steg 3: Kalibrerings- og sharpness-plot (Gauss vs. bootstrap vs. kvantil).

Skriv til: resultater/usikkerhet_*.csv og *.png
Les fraa: sarima_prognose_h*.csv, sarimax_prognose_h*.csv,
          sarima_ci_dekning.csv, ml_ensemble_prediksjoner.csv,
          laks_ukentlig_features.csv, xgboost_tunet.csv, lgbm_tunet.csv
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "004 data" / "Analyseklart datasett"
UT_DIR = Path(__file__).parent / "resultater"
UT_DIR.mkdir(exist_ok=True)

HORISONTER = [4, 8, 12]
TEST_UKER = 104
N_BOOTSTRAP = 2000
ALPHA = 0.05        # nominell 95 % CI
RANDOM_STATE = 42

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (1, 1, 1, 52)
EXOG_KOLS = ["eur_nok_snitt", "usd_nok_snitt"]

EKSKLUDER_ALLTID = {"iso_aar", "iso_uke", "uke_kode", "eksport_pris_nok_kg"}

np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Hjelpefunksjonar
# ---------------------------------------------------------------------------

def dekning_og_bredde(
    y_test: pd.Series,
    yhat_low: pd.Series,
    yhat_high: pd.Series,
) -> tuple[float, float, int]:
    """Berekn empirisk dekning og gjennomsnittleg intervallbredde."""
    felles = y_test.index.intersection(yhat_low.index)
    y  = y_test.loc[felles]
    lo = yhat_low.loc[felles]
    hi = yhat_high.loc[felles]
    innenfor = (y >= lo) & (y <= hi)
    return float(innenfor.mean()), float((hi - lo).mean()), len(felles)


def bygg_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EKSKLUDER_ALLTID]


def bygg_datasett(df: pd.DataFrame, horisont: int, feature_kols: list, cutoff: pd.Timestamp):
    y = df["eksport_pris_nok_kg"].shift(-horisont)
    data = pd.concat([df[feature_kols], y.rename("target")], axis=1).dropna()
    mask = data.index <= cutoff
    return (
        data.loc[mask, feature_kols],
        data.loc[mask, "target"],
        data.loc[~mask, feature_kols],
    )


# ---------------------------------------------------------------------------
# Steg 1: Bootstrap-CI for SARIMA / SARIMAX
# ---------------------------------------------------------------------------

def fit_sarima(y_train, exog_train=None):
    """Fit SARIMA(X) og returner resultatobjekt."""
    namn = "SARIMAX" if exog_train is not None else "SARIMA"
    print(f"  [{namn}] Tilpasser paa {len(y_train)} treningspunkter ...")
    res = SARIMAX(
        y_train,
        exog=exog_train,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    print(f"  [{namn}] AIC={res.aic:.1f}")
    return res


def hent_bootstrap_residualar(res, n_skip: int = 52) -> np.ndarray:
    """In-sample 1-steg-feil i original NOK/kg, hoppar over init-obs."""
    fitted = res.fittedvalues
    faktisk = np.asarray(res.model.endog).flatten()
    faktisk_s = pd.Series(faktisk, index=fitted.index)
    resid = (faktisk_s - fitted).iloc[n_skip:].dropna()
    print(f"    Bootstrap-residualar: n={len(resid)}, "
          f"std={resid.std():.2f}, skew={float(resid.skew()):.2f}, "
          f"kurt={float(resid.kurt()):.2f}")
    return resid.values


def bootstrap_ci_for_prognose(
    prognose_df: pd.DataFrame,
    y_test: pd.Series,
    resid: np.ndarray,
    modell: str,
    horisont: int,
) -> dict:
    """Lag bootstrap-CI for ein gitt horisont og samanlikn med Gauss-CI.

    In-sample 1-stegs residualar (std << h-stegs feil) vert skalert til
    korrekt h-stegs varians foer bootstrapping. Skaleringsfaktoren er
    sigma_h / sigma_1 der sigma_h = gauss_CI_bredde / (2 * 1.96).
    Bootstrappen bevarer den empiriske fete-hale-forma men gjev riktig
    absolutt skala -- og breiare CI enn Gauss naar kurtose > 3.
    """
    # Gauss-baseline fraa eksisterande CI
    gauss_dek, gauss_br, _ = dekning_og_bredde(
        y_test, prognose_df["yhat_low"], prognose_df["yhat_high"]
    )

    # Skaler residualar til h-stegs varians
    sigma_1 = resid.std()
    sigma_h = gauss_br / (2 * 1.96)      # implisitt sigma fraa Gauss-CI
    skala   = sigma_h / sigma_1 if sigma_1 > 0 else 1.0
    resid_skalert = resid * skala

    yhat = prognose_df["yhat"]
    felles = yhat.index.intersection(y_test.index)
    yhat = yhat.loc[felles]

    # Bootstrap: sample N_BOOTSTRAP residualar per prognose-punkt
    samples = np.random.choice(resid_skalert, size=(len(yhat), N_BOOTSTRAP), replace=True)
    boot_preds = yhat.values[:, None] + samples   # (n_test, N_BOOTSTRAP)

    lo = np.percentile(boot_preds, 100 * ALPHA / 2, axis=1)
    hi = np.percentile(boot_preds, 100 * (1 - ALPHA / 2), axis=1)

    lo_s = pd.Series(lo, index=felles)
    hi_s = pd.Series(hi, index=felles)

    dekning, bredde, n = dekning_og_bredde(y_test, lo_s, hi_s)

    print(f"    h={horisont}: bootstrap dekning={dekning:.1%} (bredde={bredde:.1f}, "
          f"skala={skala:.2f}x), Gauss={gauss_dek:.1%} (bredde={gauss_br:.1f})")

    return {
        "modell": modell + "_bootstrap",
        "horisont": horisont,
        "n": n,
        "dekning": round(dekning, 4),
        "gj_bredde": round(bredde, 4),
        "gauss_dekning": round(gauss_dek, 4),
        "gauss_bredde": round(gauss_br, 4),
        "skala_faktor": round(skala, 3),
    }


def kjor_bootstrap(df, y_train, y_test, exog_train, exog_test) -> list:
    """Kjoer bootstrap-CI for baade SARIMA og SARIMAX."""
    rader = []
    for modell, exog_tr, exog_te in [
        ("SARIMA",  None,       None),
        ("SARIMAX", exog_train, exog_test),
    ]:
        print(f"\n[Bootstrap {modell}]")
        res = fit_sarima(y_train, exog_tr)
        resid = hent_bootstrap_residualar(res)

        for h in HORISONTER:
            prog = pd.read_csv(
                UT_DIR / f"{modell.lower()}_prognose_h{h}.csv",
                parse_dates=["uke_start"],
            ).set_index("uke_start")
            rad = bootstrap_ci_for_prognose(prog, y_test, resid, modell, h)
            rader.append(rad)

    return rader


# ---------------------------------------------------------------------------
# Steg 2: Kvantilregresjon med LightGBM
# ---------------------------------------------------------------------------

def last_lgbm_params(horisont: int) -> dict:
    df = pd.read_csv(UT_DIR / "lgbm_tunet.csv")
    return json.loads(df[df["horisont"] == horisont].iloc[0]["beste_params"])


def kjor_quantile_lgbm(df, cutoff) -> list:
    """Tren LightGBM-kvantilmodellar og evaluer dekning og bredde."""
    feature_kols = bygg_features(df)
    y_test_full = df.iloc[-TEST_UKER:]["eksport_pris_nok_kg"]
    rader = []

    for h in HORISONTER:
        print(f"\n[Kvantil LightGBM h={h}]")
        X_tr, y_tr, X_te = bygg_datasett(df, h, feature_kols, cutoff)
        params = last_lgbm_params(h)
        params.pop("n_estimators", None)

        preds = {}
        for q in [0.025, 0.5, 0.975]:
            m = LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=1000,
                learning_rate=0.03,          # saktare laering for kvantil
                max_depth=params.get("max_depth", -1),
                num_leaves=params.get("num_leaves", 31),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                min_child_samples=20,        # regularisering
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            m.fit(X_tr, y_tr)
            pred = pd.Series(m.predict(X_te), index=X_te.index)
            pred.index = pred.index + pd.Timedelta(weeks=h)
            preds[q] = pred.reindex(y_test_full.index)

        lo = preds[0.025]
        hi = preds[0.975]
        mid = preds[0.5]

        dekning, bredde, n = dekning_og_bredde(y_test_full, lo, hi)

        # Sanity check: median vs. ensemble-mean
        ens_pred = pd.read_csv(
            UT_DIR / "ml_ensemble_prediksjoner.csv", parse_dates=["uke_start"]
        ).set_index("uke_start")
        ens_h = ens_pred[ens_pred["horisont"] == h]["ensemble_pred"].reindex(y_test_full.index)
        felles = mid.dropna().index.intersection(ens_h.dropna().index)
        diff_median_ens = (mid.loc[felles] - ens_h.loc[felles]).abs().mean()

        print(f"    h={h}: dekning={dekning:.1%}, bredde={bredde:.1f} NOK/kg")
        print(f"    Median vs. ensemble MAE={diff_median_ens:.2f} (sanity)")

        rader.append({
            "modell": "LightGBM_quantile",
            "horisont": h,
            "n": n,
            "dekning": round(dekning, 4),
            "gj_bredde": round(bredde, 4),
        })

    return rader


# ---------------------------------------------------------------------------
# Steg 3: Kalibreringsplot og sharpness
# ---------------------------------------------------------------------------

def bygg_kalibrering_df(gauss_df, bootstrap_rader, quantile_rader) -> pd.DataFrame:
    """Saml alle metodar i ein tabell for samanlikning."""
    rader = []

    for _, r in gauss_df.iterrows():
        rader.append({
            "modell": r["modell"] + "_gauss",
            "horisont": int(r["horisont"]),
            "dekning": r["dekning"],
            "gj_bredde": r["gj_bredde"],
            "nominell": r["nominell"],
        })

    for r in bootstrap_rader:
        rader.append({
            "modell": r["modell"],
            "horisont": r["horisont"],
            "dekning": r["dekning"],
            "gj_bredde": r["gj_bredde"],
            "nominell": 1 - ALPHA,
        })

    for r in quantile_rader:
        rader.append({
            "modell": r["modell"],
            "horisont": r["horisont"],
            "dekning": r["dekning"],
            "gj_bredde": r["gj_bredde"],
            "nominell": 1 - ALPHA,
        })

    return pd.DataFrame(rader)


def plot_kalibrering(kal_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    stiler = {
        "SARIMA_gauss":       ("tab:orange",  "o", "--"),
        "SARIMAX_gauss":      ("tab:green",   "s", "--"),
        "SARIMA_bootstrap":   ("tab:orange",  "o", "-"),
        "SARIMAX_bootstrap":  ("tab:green",   "s", "-"),
        "LightGBM_quantile":  ("tab:blue",    "^", "-"),
    }

    for ax, h in zip(axes, HORISONTER):
        sub = kal_df[kal_df["horisont"] == h]
        ax.axline((0, 0), slope=1, color="black", lw=0.8, linestyle=":", label="Perfekt kalibrert")
        for _, row in sub.iterrows():
            farge, markør, stil = stiler.get(row["modell"], ("gray", "x", "-"))
            ax.scatter(
                row["nominell"], row["dekning"],
                color=farge, marker=markør, s=80, zorder=5,
            )
            ax.annotate(
                row["modell"].replace("_", "\n"),
                (row["nominell"], row["dekning"]),
                textcoords="offset points", xytext=(5, 3), fontsize=6,
            )
        ax.set_xlim(0.7, 1.0)
        ax.set_ylim(0.7, 1.0)
        ax.set_xlabel("Nominell dekning")
        ax.set_title(f"h={h} veker")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Empirisk dekning")
    plt.suptitle("Kalibrering: empirisk vs. nominell dekning (maal: paa y=x-linja)", fontsize=11)
    plt.tight_layout()
    fig.savefig(UT_DIR / "usikkerhet_kalibrering.png", dpi=120)
    plt.close(fig)
    print("  Lagra: usikkerhet_kalibrering.png")


def plot_sharpness(kal_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fargar = {
        "SARIMA_gauss":       "tab:orange",
        "SARIMAX_gauss":      "tab:green",
        "SARIMA_bootstrap":   "tab:red",
        "SARIMAX_bootstrap":  "tab:purple",
        "LightGBM_quantile":  "tab:blue",
    }

    for ax, h in zip(axes, HORISONTER):
        sub = kal_df[kal_df["horisont"] == h].copy()
        sub = sub.sort_values("gj_bredde")
        fargeliste = [fargar.get(m, "gray") for m in sub["modell"]]
        ax.barh(sub["modell"], sub["gj_bredde"], color=fargeliste)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(
                row["gj_bredde"] + 0.3, i,
                f'{row["dekning"]:.0%}',
                va="center", fontsize=8,
            )
        ax.set_xlabel("Gj.snittleg CI-bredde (NOK/kg)")
        ax.set_title(f"h={h} veker  [tal = empirisk dekning]")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Sharpness: smalare er betre -- gitt god kalibrering", fontsize=11)
    plt.tight_layout()
    fig.savefig(UT_DIR / "usikkerhet_sharpness.png", dpi=120)
    plt.close(fig)
    print("  Lagra: usikkerhet_sharpness.png")


# ---------------------------------------------------------------------------
# Hovudprogram
# ---------------------------------------------------------------------------

def main() -> None:
    df = (
        pd.read_csv(DATA_DIR / "laks_ukentlig_features.csv", parse_dates=["uke_start"])
        .set_index("uke_start")
        .sort_index()
    )
    train = df.iloc[:-TEST_UKER]
    test  = df.iloc[-TEST_UKER:]
    cutoff = train.index.max()

    y_train = train["eksport_pris_nok_kg"].asfreq("W-MON")
    y_test  = test["eksport_pris_nok_kg"].asfreq("W-MON")
    exog_train = train[EXOG_KOLS].asfreq("W-MON")
    exog_test  = test[EXOG_KOLS].asfreq("W-MON")

    print(f"Train: {train.index.min().date()} - {train.index.max().date()} ({len(train)} uker)")
    print(f"Test:  {test.index.min().date()} - {test.index.max().date()} ({len(test)} uker)")

    # Les Gauss-baseline fraa Spor A
    gauss_df = pd.read_csv(UT_DIR / "sarima_ci_dekning.csv")

    # -----------------------------------------------------------------------
    # Steg 1: Bootstrap-CI
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEG 1: Bootstrap-CI for SARIMA og SARIMAX")
    print("=" * 60)

    bootstrap_rader = kjor_bootstrap(df, y_train, y_test, exog_train, exog_test)

    boot_df = pd.DataFrame(bootstrap_rader)
    boot_df.to_csv(UT_DIR / "usikkerhet_sarima_bootstrap.csv", index=False)
    print("\n--- Bootstrap-CI ---")
    print(boot_df[["modell", "horisont", "n", "dekning", "gj_bredde",
                   "gauss_dekning", "gauss_bredde"]].to_string(index=False))

    # -----------------------------------------------------------------------
    # Steg 2: Kvantilregresjon LightGBM
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEG 2: Kvantilregresjon LightGBM")
    print("=" * 60)

    quantile_rader = kjor_quantile_lgbm(df, cutoff)

    quant_df = pd.DataFrame(quantile_rader)
    quant_df.to_csv(UT_DIR / "usikkerhet_ml_quantile.csv", index=False)
    print("\n--- Kvantil LightGBM ---")
    print(quant_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Steg 3: Samanlikningstabell og plot
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEG 3: Kalibrering og sharpness")
    print("=" * 60)

    kal_df = bygg_kalibrering_df(gauss_df, bootstrap_rader, quantile_rader)
    kal_df.to_csv(UT_DIR / "usikkerhet_kalibrering.csv", index=False)

    print("\nLagar figurer ...")
    plot_kalibrering(kal_df)
    plot_sharpness(kal_df)

    # -----------------------------------------------------------------------
    # Oppsummering
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OPPSUMMERING - Spor G")
    print("=" * 60)
    print(f"{'Modell':<25} {'h=4 dek':>9} {'h=8 dek':>9} {'h=12 dek':>10}")
    print("-" * 55)
    for modell in kal_df["modell"].unique():
        sub = kal_df[kal_df["modell"] == modell].set_index("horisont")
        dekningar = [f"{sub.loc[h, 'dekning']:.1%}" if h in sub.index else "  -  "
                     for h in HORISONTER]
        print(f"  {modell:<23} {dekningar[0]:>9} {dekningar[1]:>9} {dekningar[2]:>10}")

    print(f"\nMaal: dekning innanfor 92-98 % (nominell 95 %)")
    print(f"Resultater lagra i {UT_DIR}")


if __name__ == "__main__":
    main()
