# Analyseklart datasett – ukentlig

Sammenslått tidsserie for prediksjon av eksportpris på fersk norsk laks.

## Frekvens og tidsindeks

- **Ukentlig**, ISO-uke (mandag som første dag).
- Periode: **2010-01-04 til 2026-03-09** (845 uker).
- Tidsindeks i filene:
  - `iso_aar`, `iso_uke` – ISO-år og uke (heltall)
  - `uke_start` – mandag i ISO-uken (datotype)
  - `uke_kode` – SSB-format `YYYYUWW` (f.eks. `2026U11`)

## Filer

| Fil | Innhold |
|---|---|
| `bygg_datasett.py` | Skriptet som bygger filene fra rådataene i `004 data/` |
| `laks_ukentlig.csv` | Rent sammenslått datasett (13 kolonner) |
| `laks_ukentlig_features.csv` | Som over + lag-, sesong- og endrings-variabler (32 kolonner). **Bruk denne til modellering.** |

## Kolonner

### Basis (`laks_ukentlig.csv`)

| Kolonne | Kilde | Beskrivelse |
|---|---|---|
| `eksport_volum_tonn` | SSB tab. 03024 | Ukentlig eksportert volum, tonn |
| `eksport_pris_nok_kg` | SSB tab. 03024 | Ukentlig eksportpris, NOK/kg (**målvariabel**) |
| `eur_nok_snitt` / `eur_nok_ukeslutt` | Norges Bank | EUR/NOK – snitt for uken / siste virkedag |
| `usd_nok_snitt` / `usd_nok_ukeslutt` | Norges Bank | USD/NOK – snitt / siste virkedag |
| `fao_global_atlantisk_tonn` | FAO | Global produksjon av atlantisk laks (årlig, broadcastet til alle uker i året) |
| `fao_norge_tonn` | FAO | Norsk del av samme |
| `fao_eks_norge_tonn` | FAO | Global produksjon **eksklusiv Norge** |

### Tilleggskolonner (`laks_ukentlig_features.csv`)

- **Lag-priser:** `pris_lag_1`, `pris_lag_2`, `pris_lag_4`, `pris_lag_8`, `pris_lag_12`
- **Lag-volum:** `volum_lag_1`, `volum_lag_4`, `volum_lag_12`
- **Glidende snitt pris:** `pris_ma_4`, `pris_ma_12` (lag-1-basert, ingen lekkasje)
- **Prisendringer:** `pris_endring_1u`, `pris_endring_4u`, `pris_endring_52u`
- **Valutaendringer:** `eur_endring_4u`, `usd_endring_4u`
- **Sesong:** `maaned`, `kvartal`, `uke_sin`, `uke_cos` (syklisk koding av uke)
- **Volum-differanser (Spor C, 2026-04-29):** `volum_endring_1u`, `volum_endring_4u`, `volum_endring_52u` (lag-1-basert)
- **Valutaratio (Spor C, 2026-04-29):** `eur_usd_ratio = eur_nok_snitt / usd_nok_snitt`
- **Akkumulert volum (Spor C, 2026-04-29):** `volum_sum_4u`, `volum_sum_12u`, `volum_sum_52u` (lag-1-basert rullende sum)
- **Glidende snitt volum (Spor C, 2026-04-29):** `volum_ma_4`, `volum_ma_12` (lag-1-basert)
- **Prisvolatilitet (Spor C, 2026-04-29):** `pris_std_4`, `pris_std_12` (lag-1-basert rullende std)
- **Spot vs. trend (Spor C, 2026-04-29):** `pris_vs_ma_12 = pris_lag_1 / pris_ma_12`
- **FAO-imputation (Spor C, 2026-04-29):** `fao_imputert` – binært flagg (1 = forward-fill frå 2022-verdi, 0 = observert). FAO-kolonnene er no NaN-frie for heile perioden.

## Kjente begrensninger

- **FAO-data slutter i 2022.** 167 uker i 2023–2026 mangler verdier for `fao_*`-kolonnene. Tre alternativer ved modellering:
  1. forward-fill 2022-verdien
  2. ekstrapoler trend
  3. dropp FAO-kolonnene ved trening på de siste årene
- Lag-baserte features har `NaN` i starten av serien – dropp første ~52 rader før trening.
- Norges Bank-data går til 2026-03-20; SSB-data til 2026U11 (uke som starter 2026-03-09). Kombinert datasett er kuttet ved siste komplette SSB-uke.
- Aggregering valuta → uke bruker både snitt og siste virkedag (typisk fredag). Velg den varianten som gir best resultat under modellutvelgelse.

## Reproduksjon

```bash
cd "004 data/Analyseklart datasett"
python bygg_datasett.py
```

Krever `pandas` og `openpyxl`. Skriptet leser fra `../Fra Teams/` og `../AtlanticSalmon_GlobalProduction_2000_2022.xlsx`.
