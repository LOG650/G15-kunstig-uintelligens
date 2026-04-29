"""
Bygger sammenslatt ukentlig datasett for prisprediksjon pa fersk norsk laks.

Kilder:
  - SSB tabell 03024: ukentlig eksportpris (NOK/kg) og volum (tonn) for fersk oppdrettslaks
  - Norges Bank: daglige EUR/NOK og USD/NOK valutakurser
  - FAO: arlig global produksjon av atlantisk laks per land

Output (ISO ar-uke som tidsindeks):
  - laks_ukentlig.csv          (ren sammenslatt data)
  - laks_ukentlig_features.csv (med lag- og sesongvariabler for modellering)
"""

import math
from datetime import date
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # mappen "004 data"
TEAMS = ROOT / "Fra Teams"
OUT = Path(__file__).resolve().parent

SSB_FILE = TEAMS / "eksport 2010 - 26.csv"
EUR_FILE = TEAMS / "nok_eur.csv.csv"
USD_FILE = TEAMS / "nok_usd.csv.csv"
FAO_FILE = ROOT / "AtlanticSalmon_GlobalProduction_2000_2022.xlsx"


def les_ssb_eksport(path: Path) -> pd.DataFrame:
    """SSB tabell 03024. Forste linje er tittel, andre linje er header."""
    df = pd.read_csv(path, sep=";", skiprows=1, encoding="utf-8")
    df.columns = ["uke_kode", "eksport_volum_tonn", "eksport_pris_nok_kg"]
    df[["iso_aar", "iso_uke"]] = df["uke_kode"].str.extract(r"(\d{4})U(\d{2})").astype(int)
    df["uke_start"] = df.apply(
        lambda r: date.fromisocalendar(r["iso_aar"], r["iso_uke"], 1), axis=1
    )
    df["uke_start"] = pd.to_datetime(df["uke_start"])
    return df[["iso_aar", "iso_uke", "uke_start", "uke_kode",
               "eksport_volum_tonn", "eksport_pris_nok_kg"]]


def les_norges_bank(path: Path, kvote: str) -> pd.DataFrame:
    """Norges Bank SDMX-eksport, semikolon-separert, komma som desimaltegn."""
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["dato", f"{kvote}_nok"]
    df["dato"] = pd.to_datetime(df["dato"])
    return df


def aggreger_valuta_til_uke(fx_daily: pd.DataFrame, kvote: str) -> pd.DataFrame:
    """Snitt + ukens slutt (siste virkedag) per ISO ar-uke."""
    kol = f"{kvote}_nok"
    iso = fx_daily["dato"].dt.isocalendar()
    fx = fx_daily.assign(iso_aar=iso["year"].astype(int), iso_uke=iso["week"].astype(int))
    fx = fx.sort_values("dato")
    agg = fx.groupby(["iso_aar", "iso_uke"], as_index=False).agg(
        **{
            f"{kvote}_nok_snitt": (kol, "mean"),
            f"{kvote}_nok_ukeslutt": (kol, "last"),
        }
    )
    return agg


def les_fao_atlantisk_laks(path: Path) -> pd.DataFrame:
    """Globalt tilbud per ar (atlantisk laks). Brukes som arlig variabel."""
    df = pd.read_excel(path, sheet_name="Globalt tilbud per år")
    df = df.rename(columns={
        "Year": "iso_aar",
        "Global total": "fao_global_atlantisk_tonn",
        "Norway": "fao_norge_tonn",
    })
    df["fao_eks_norge_tonn"] = df["fao_global_atlantisk_tonn"] - df["fao_norge_tonn"]
    return df[["iso_aar", "fao_global_atlantisk_tonn",
               "fao_norge_tonn", "fao_eks_norge_tonn"]]


def bygg_basisdatasett() -> pd.DataFrame:
    ssb = les_ssb_eksport(SSB_FILE)
    eur = aggreger_valuta_til_uke(les_norges_bank(EUR_FILE, "eur"), "eur")
    usd = aggreger_valuta_til_uke(les_norges_bank(USD_FILE, "usd"), "usd")
    fao = les_fao_atlantisk_laks(FAO_FILE)

    df = ssb.merge(eur, on=["iso_aar", "iso_uke"], how="left")
    df = df.merge(usd, on=["iso_aar", "iso_uke"], how="left")
    df = df.merge(fao, on="iso_aar", how="left")
    return df.sort_values("uke_start").reset_index(drop=True)


def legg_til_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for k in [1, 2, 4, 8, 12]:
        out[f"pris_lag_{k}"] = out["eksport_pris_nok_kg"].shift(k)

    for k in [1, 4, 12]:
        out[f"volum_lag_{k}"] = out["eksport_volum_tonn"].shift(k)

    out["pris_ma_4"] = out["eksport_pris_nok_kg"].shift(1).rolling(4).mean()
    out["pris_ma_12"] = out["eksport_pris_nok_kg"].shift(1).rolling(12).mean()
    out["pris_endring_1u"] = out["eksport_pris_nok_kg"].pct_change(1)
    out["pris_endring_4u"] = out["eksport_pris_nok_kg"].pct_change(4)
    out["pris_endring_52u"] = out["eksport_pris_nok_kg"].pct_change(52)

    out["eur_endring_4u"] = out["eur_nok_ukeslutt"].pct_change(4)
    out["usd_endring_4u"] = out["usd_nok_ukeslutt"].pct_change(4)

    out["maaned"] = out["uke_start"].dt.month
    out["kvartal"] = out["uke_start"].dt.quarter
    radian = out["iso_uke"] / 52 * 2 * math.pi
    out["uke_sin"] = radian.apply(math.sin)
    out["uke_cos"] = radian.apply(math.cos)

    # --- Spor C: nye features (alle lag-1-basert for å unngå leakage) ---

    # 1. Volum-differanser: shift(1) gir volum[t-1] ved tidspunkt t;
    #    pct_change(k) beregnar endringa over k periodar tilbake frå t-1.
    volum_s1 = out["eksport_volum_tonn"].shift(1)
    out["volum_endring_1u"] = volum_s1.pct_change(1)    # (volum[t-1]-volum[t-2])/volum[t-2]
    out["volum_endring_4u"] = volum_s1.pct_change(4)    # (volum[t-1]-volum[t-5])/volum[t-5]
    out["volum_endring_52u"] = volum_s1.pct_change(52)  # (volum[t-1]-volum[t-53])/volum[t-53]

    # 2. EUR/USD-ratio: spotverdi kjent ved t (ikkje framtidsdata),
    #    analogt med eksisterande eur_endring_4u / usd_endring_4u.
    out["eur_usd_ratio"] = out["eur_nok_snitt"] / out["usd_nok_snitt"]

    # 3. Akkumulert volum: shift(1) + rolling sum → sum av volum[t-k..t-1]
    out["volum_sum_4u"] = volum_s1.rolling(4).sum()
    out["volum_sum_12u"] = volum_s1.rolling(12).sum()
    out["volum_sum_52u"] = volum_s1.rolling(52).sum()

    # 4. Glidende snitt volum: lag-1-basert, analogt med pris_ma_*
    out["volum_ma_4"] = volum_s1.rolling(4).mean()
    out["volum_ma_12"] = volum_s1.rolling(12).mean()

    # 5. Prisvolatilitet: shift(1) + rolling std → std av pris[t-k..t-1]
    pris_s1 = out["eksport_pris_nok_kg"].shift(1)
    out["pris_std_4"] = pris_s1.rolling(4).std()
    out["pris_std_12"] = pris_s1.rolling(12).std()

    # 6. Spot vs. langsiktig trend: begge inngangsverdiar er lag-1-baserte
    out["pris_vs_ma_12"] = out["pris_lag_1"] / out["pris_ma_12"]

    return out


def main() -> None:
    base = bygg_basisdatasett()
    base_path = OUT / "laks_ukentlig.csv"
    base.to_csv(base_path, index=False, encoding="utf-8")
    print(f"Skrev {base_path.name}: {len(base)} rader, {len(base.columns)} kolonner")
    print(f"  Periode: {base['uke_start'].min().date()} til {base['uke_start'].max().date()}")

    features = legg_til_features(base)
    feat_path = OUT / "laks_ukentlig_features.csv"
    features.to_csv(feat_path, index=False, encoding="utf-8")
    print(f"Skrev {feat_path.name}: {len(features)} rader, {len(features.columns)} kolonner")

    mangler = base[["eur_nok_ukeslutt", "usd_nok_ukeslutt",
                    "fao_global_atlantisk_tonn"]].isna().sum()
    print("\nManglende verdier i basis (etter merge):")
    print(mangler.to_string())

    # Leakage-verifisering: ved indeks i skal feature berre bruke data t.o.m. i-1
    i = 60  # etter warm-up-perioden (>52 rader)
    pris = features["eksport_pris_nok_kg"]
    volum = features["eksport_volum_tonn"]

    exp_vendring = (volum.iloc[i - 1] - volum.iloc[i - 2]) / volum.iloc[i - 2]
    assert abs(features["volum_endring_1u"].iloc[i] - exp_vendring) < 1e-9, \
        "Leakage: volum_endring_1u bruker data etter t-1"

    assert abs(features["volum_sum_4u"].iloc[i] - volum.iloc[i - 4:i].sum()) < 1e-9, \
        "Leakage: volum_sum_4u bruker data etter t-1"

    assert abs(features["volum_ma_4"].iloc[i] - volum.iloc[i - 4:i].mean()) < 1e-9, \
        "Leakage: volum_ma_4 bruker data etter t-1"

    assert abs(features["pris_std_4"].iloc[i] - pris.iloc[i - 4:i].std(ddof=1)) < 1e-9, \
        "Leakage: pris_std_4 bruker data etter t-1"

    assert abs(features["pris_std_12"].iloc[i] - pris.iloc[i - 12:i].std(ddof=1)) < 1e-9, \
        "Leakage: pris_std_12 bruker data etter t-1"

    print("\nLeakage-assert OK – alle nye features bruker berre t-1 og tidlegare.")


if __name__ == "__main__":
    main()
