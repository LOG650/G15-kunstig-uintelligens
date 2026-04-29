# Analyse – baseline-modeller og videre arbeid

Status og overlevering for prediksjon av eksportpris fersk norsk laks.

**Sist oppdatert:** 2026-04-29 (Spor B)

## Filer i denne mappa

| Fil | Innhold |
|---|---|
| `baseline_modeller.ipynb` | Hovednotebook: laster data, trener Naiv/SARIMA/XGBoost, plotter. **Frosset som referanse – ikke endre** |
| `eksporter_baseline.py` | Headless-versjon av samme logikk; skriver metrikker til `resultater/`. **Frosset – ikke endre** |
| `_bygg_notebook.py` | Hjelpeskript brukt til å generere notebook-skjelettet |
| `resultater/baseline_metrikker.csv` | Long-format MAE/MAPE per (modell, horisont) |
| `resultater/baseline_metrikker_pivot.csv` | Samme tall, pivotert for rapport |
| `oppgaver/spor_*.md` | Selvstendige prompts per arbeidsspor – lim inn i AI-en din |
| `sarima_*.py / .ipynb` | Spor A – SARIMA (opprettes av Spor A) |
| `ml_*.py / .ipynb` | Spor B – XGBoost/LightGBM (opprettes av Spor B) |

## Parallell arbeidsfordeling (3 personer)

Tre spor jobber samtidig uten å rope hverandre over ende. Hvert spor har egen prompt under [`oppgaver/`](oppgaver/):

| Spor | Tema | Eier filer | Prompt |
|---|---|---|---|
| **A** | SARIMA / SARIMAX / KI | `006 analyse/sarima_*` | [`oppgaver/spor_a_sarima.md`](oppgaver/spor_a_sarima.md) |
| **B** | XGBoost / LightGBM / tuning | `006 analyse/ml_*` | [`oppgaver/spor_b_ml.md`](oppgaver/spor_b_ml.md) |
| **C** | Feature engineering | `004 data/Analyseklart datasett/bygg_datasett.py` + features-CSV | [`oppgaver/spor_c_features.md`](oppgaver/spor_c_features.md) |

Konfliktregler:
- `baseline_modeller.ipynb` og `eksporter_baseline.py` er FROSSET. Ingen endrer disse.
- Spor C kan kun **legge til** kolonner i features-CSV, aldri rename eller slette.
- Alle skriver resultater til `resultater/<eget-prefix>_*.csv`. Aldri overskriv andres filer.
- LES_MEG.md (denne fila) oppdateres av alle, men i ulike seksjoner. Tabellen og listene er additive – konflikter løses ved å beholde begge.

## Reproduksjon

```bash
cd "006 analyse"
# Alternativ 1: kjor notebooken (krever nbconvert)
python -m nbconvert --to notebook --execute baseline_modeller.ipynb \
    --output baseline_modeller.ipynb --ExecutePreprocessor.timeout=600

# Alternativ 2: bare metrikker, ingen plot
python eksporter_baseline.py
```

Krever `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `xgboost`, `matplotlib`.
Datasettet leses fra `../004 data/Analyseklart datasett/laks_ukentlig_features.csv`.

## Oppsett

- **Testset:** siste 104 uker (~2 år), kronologisk split
- **Horisonter:** 4, 8, 12 uker
- **Metrikker:** MAE, MAPE
- FAO-kolonnene droppes fra XGBoost-features fordi de er `NaN` fra 2023 – se [LES_MEG i datamappa](../004%20data/Analyseklart%20datasett/LES_MEG.md)

## Resultater per 2026-04-29

| Horisont | Naiv MAE | Naiv MAPE | SARIMA MAE | SARIMA MAPE | XGBoost (baseline) MAE | XGBoost (tunet) MAE | LightGBM (tunet) MAE | Ensemble+ES MAE | Beste ML MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | **8.51** | **9.8 %** | 22.19 | 27.9 % | 11.46 | 10.37 | 10.45 | 9.86 | 9.86 |
| 8 | **13.04** | **15.4 %** | 22.19 | 27.9 % | 12.59 | 11.98 | **10.90** | 11.38 | **10.90** |
| 12 | 16.35 | 19.7 % | 22.19 | 27.9 % | 14.79 | 15.47 | 13.06 | 13.55 | **12.66** |

Tuning: `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=5)`, 60 iterasjoner, MAE-scoring, FAO-kolonner ekskludert. Ensemble: uvektet gjennomsnitt XGBoost+LightGBM med early stopping (ES-val = siste 52 uker av treningssett). Beste ML h=12 er LightGBM+ES (12.66).

**Tolkning:**

- ML slår naiv på h=8 og h=12 – best er LightGBM (10.90) og LightGBM+ES (12.66).
- h=4: naiv (8.51) holder fortsatt. Early stopping + ensemble reduserte MAE fra 10.37 til 9.86, men gapet (~1.35 NOK/kg) er sannsynligvis fundamentalt – kortsiktige prisserier domineres av siste kjente pris.
- SARIMA gir samme tall på tvers av horisonter – se kjent problem nedenfor.

## Kjente problemer (må fikses før videre modellering)

1. **SARIMA-evalueringen er ikke fair.** Notebooken gjør single-fit + multi-step forecast,
   så samme prognoseserie sammenlignes for alle horisonter. Skal erstattes med
   rullende-opphav (refit eller `extend` for hver uke i testperioden, hent ut
   prediksjonen `h` skritt fram). Se kommentar i celle 6 i notebooken.
2. **FAO-data dropping er en proxy.** Kolonnene fjernes helt fordi 2023+ er `NaN`.
   Forward-fill 2022-verdien + en `fao_imputert`-flagg lar oss beholde signalet
   uten leakage. Beslutning ikke tatt.
3. **Lag-features har NaN i de første ~52 radene.** Notebooken dropper dem implisitt
   via `dropna()` i XGBoost-blokken, men ikke for naiv/SARIMA. Greit nå, men
   sammenligningsgrunnlag (`n` per modell) bør sjekkes ved rapport.

## Neste steg (prioritert)

1. **Fiks SARIMA-evalueringen** med rullende opphav. Dette er forutsetningen for
   at sammenligningen mot XGBoost skal være meningsfull.
2. ~~**Bestem FAO-håndtering**~~ **Ferdig (Spor C, 2026-04-29)** – FAO-kolonnene er forward-filla for 2023–2026; `fao_imputert`-flagg skiller observerte fra interpolerte rader. XGBoost kan no bruke FAO-features utan å droppe dei.
3. ~~**Hyperparameter-tuning XGBoost** (`RandomizedSearchCV` + `TimeSeriesSplit`)~~ **Ferdig (Spor B, 2026-04-29)** – tunet XGBoost og LightGBM, resultat i `resultater/xgboost_tunet.csv` og `lgbm_tunet.csv`.
4. **SARIMAX med valuta som eksogen regressor** (`eur_nok_snitt`, `usd_nok_snitt`).
5. ~~**LightGBM** som alternativ ML-modell (nevnt i prosjektplanen).~~ **Ferdig (Spor B, 2026-04-29)** – LightGBM slår naiv på h=8 og h=12.
6. **Konfidensintervaller** for SARIMA-prognoser i sluttrapport.
7. ~~**Ekstra feature engineering**: differanser, EUR/USD-ratio, akkumulert volum.~~ **Ferdig (Spor C, 2026-04-29)** – 12 nye kolonner tilgjengelig i `laks_ukentlig_features.csv` (se datamappa LES_MEG).

## Tilstand i notebooken

Alle celler kjører uten feil per 2026-04-29. Outputs (tabeller, plot, feature
importance) er lagret i `baseline_modeller.ipynb`.
