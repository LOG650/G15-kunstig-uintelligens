# Analyse – baseline-modeller og videre arbeid

Status og overlevering for prediksjon av eksportpris fersk norsk laks.

<<<<<<< HEAD
**Sist oppdatert:** 2026-04-29 (Spor A ferdig)
=======
**Sist oppdatert:** 2026-04-29 (Spor B, full retren med alle Spor C-features + FAO)
>>>>>>> b146ea7f6f5a2f661933164502f9eab3c9613ad8

## Filer i denne mappa

| Fil | Innhold |
|---|---|
| `baseline_modeller.ipynb` | Hovednotebook: laster data, trener Naiv/SARIMA/XGBoost, plotter. **Frosset som referanse – ikke endre** |
| `eksporter_baseline.py` | Headless-versjon av samme logikk; skriver metrikker til `resultater/`. **Frosset – ikke endre** |
| `_bygg_notebook.py` | Hjelpeskript brukt til å generere notebook-skjelettet |
| `resultater/baseline_metrikker.csv` | Long-format MAE/MAPE per (modell, horisont) |
| `resultater/baseline_metrikker_pivot.csv` | Samme tall, pivotert for rapport |
| `oppgaver/spor_*.md` | Selvstendige prompts per arbeidsspor – lim inn i AI-en din |
<<<<<<< HEAD
| `sarima_eksperiment.py` | Spor A – SARIMA / SARIMAX med rullende opphav og 95 % CI |
| `resultater/sarima_metrikker.csv` | MAE/MAPE for SARIMA og SARIMAX per horisont |
| `resultater/sarima_prognose_h{4,8,12}.csv` | SARIMA-prognoser med 95 % CI |
| `resultater/sarimax_prognose_h{4,8,12}.csv` | SARIMAX-prognoser med 95 % CI |
| `resultater/sarima_prognoser.png` | Plot av SARIMA og SARIMAX mot fasit per horisont |
| `resultater/sarima_ci_dekning.csv` | Empirisk dekning av 95 %-CI (mål: 0.95) |
| `resultater/sarima_residualdiagnostikk.csv` | Ljung-Box + Jarque-Bera på treningsresidualer |
| `resultater/sarima_residualer.png`, `sarimax_residualer.png` | ACF og QQ-plot av standardiserte residualer |
| `ml_*.py / .ipynb` | Spor B – XGBoost/LightGBM (opprettes av Spor B) |
=======
| `sarima_*.py / .ipynb` | Spor A – SARIMA (opprettes av Spor A) |
| `ml_eksperiment.py` | Spor B – FAO-samanlikning, XGBoost/LightGBM-tuning, feature importance |
| `ml_ensemble.py` | Spor B – Ensemble XGBoost + LightGBM med early stopping; lagrar `ml_ensemble_prediksjoner.csv` |
| `ml_residualplot.py` | Spor B – Residualanalyse; les `ml_ensemble_prediksjoner.csv`, produserer 3 plotfiler |
| `resultater/ml_ensemble_prediksjoner.csv` | Faktiske og predikerte prisar per veke, horisont og modell |
| `resultater/residualplot_tid.png` | Residualar over tid per horisont (Ensemble) |
| `resultater/residualplot_scatter.png` | Faktisk vs. predikert scatter per horisont (Ensemble) |
| `resultater/residualplot_hist.png` | Histogram av residualar per horisont (Ensemble) |
>>>>>>> b146ea7f6f5a2f661933164502f9eab3c9613ad8

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

## Resultater per 2026-04-29 (full retren – alle Spor C-features + FAO inkludert)

<<<<<<< HEAD
| Horisont | Naiv MAE | Naiv MAPE | SARIMA MAE¹ | SARIMA MAPE¹ | SARIMAX MAE¹ | SARIMAX MAPE¹ | XGBoost MAE | XGBoost MAPE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8.51 | 9.8 % | **8.27** | **9.5 %** | 8.33 | 9.6 % | 11.46 | 13.7 % |
| 8 | 13.04 | 15.4 % | **11.07** | **13.1 %** | 11.07 | 13.1 % | 12.59 | 15.1 % |
| 12 | 16.35 | 19.7 % | 13.15 | 15.9 % | **12.93** | **15.6 %** | 14.79 | 18.1 % |

¹ Rullende-opphav-evaluering (refit=False per uke). SARIMAX bruker `eur_nok_snitt` og `usd_nok_snitt` som eksogene regressorer. AIC for begge ≈ 3222 på treningsdata.

**Tolkning:**
- SARIMA slår naiv på alle tre horisontene etter at evalueringen er gjort fair.
- SARIMAX gir marginal gevinst kun på h=12; valutaeffekten er allerede delvis fanget av lag-strukturen.
- XGBoost (utuned) er fortsatt den svakeste modellen på h=4 og h=8; tuning og bedre features ventes å løfte den (Spor B og C).
- Prognoser med 95 % CI er lagret per modell og horisont i `resultater/`.

### CI-kalibrering og residualdiagnostikk

Empirisk dekning av 95 %-konfidensintervallene:

| Modell | h=4 | h=8 | h=12 |
|---|---:|---:|---:|
| SARIMA | 79 % | 80 % | 81 % |
| SARIMAX | 81 % | 81 % | 82 % |

CI-ene er **systematisk for smale** (~80 % faktisk dekning vs. 95 % nominell). Dette er konsistent med residualdiagnostikken på treningsdata (standardiserte residualer):

| Modell | LB(10) p | LB(20) p | LB(52) p | JB skew | JB kurtose |
|---|---:|---:|---:|---:|---:|
| SARIMA  | 0.004 | 0.002 | <0.001 | +0.24 | 4.55 |
| SARIMAX | 0.009 | 0.004 | <0.001 | +0.28 | 4.44 |

- **Ljung-Box forkaster hvit-støy-hypotesen på alle lag** – modellen fanger ikke all autokorrelasjon, særlig sesongleddet ved lag 52.
- **Residualene har feterehaler enn normal** (kurtose ≈ 4.5 vs. 3 for normalfordeling) – derfor er Gauss-baserte CI-er for trange.
- For rapporten bør disse begrensningene rapporteres ærlig: punktprognoser er rimelige, men intervallene gir falsk presisjon.
=======
| Horisont | Naiv MAE | Naiv MAPE | SARIMA MAE | SARIMA MAPE | XGBoost (baseline) MAE | XGBoost (tunet) MAE | LightGBM (tunet) MAE | Ensemble+ES MAE | Beste ML MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8.51 | 9.8 % | 22.19 | 27.9 % | 11.46 | 10.37 | 10.45 | **8.33** | **8.33** |
| 8 | **13.04** | **15.4 %** | 22.19 | 27.9 % | 12.59 | 11.98 | 10.90 | **10.85** | **10.85** |
| 12 | 16.35 | 19.7 % | 22.19 | 27.9 % | 14.79 | 15.47 | 13.06 | **13.56** | **12.66*** |

\* h=12: XGBoost+ES (15.31) og LightGBM+ES (13.24) — ensemble (13.56) er svakare enn LightGBM aleine. Sjå note under.

Tuning: `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=5)`, 60 iterasjoner, MAE-scoring, 36 features (uten FAO; FAO+uten-FAO gav uavgjort i snitt-MAE 12.183 vs 12.184). Ensemble: uvektet gjennomsnitt XGBoost+ES + LightGBM+ES, trent med **40 features inkl. FAO** (early stopping eliminerer overfitting frå FAO-kolonner).

**Hovudfunn etter full retren:**

- **Ensemble slår naiv på alle tre horisontar** — h=4 MAE 8.33 < naiv 8.51. Dette var ikkje tilfellet i tidlegare køyringar.
- h=4-forbetringa kjem frå kombinasjonen FAO-features + early stopping (ES-val stoppar tidleg: 17–37 iterasjonar).
- **Systematisk negativ bias:** modellen underpredikerer konsekvent med ~2.2–2.9 NOK/kg på tvers av alle horisontar. Sjå residualplot.
- h=12: LightGBM åleine (13.24) slår ensemble (13.56) — ensemble-averaging dreg opp fordi XGBoost+ES er svakare på lang horisont.
- SARIMA gir same tall på tvers av horisontar – sjå kjent problem nedenfor.

**Residualanalyse (Ensemble):**

| Horisont | n | MAE | Bias | Std |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 100 | 8.33 | -2.16 | 10.56 |
| 8 | 96 | 10.85 | -2.92 | 13.44 |
| 12 | 92 | 13.56 | -2.73 | 17.03 |

Plot: `resultater/residualplot_tid.png`, `residualplot_scatter.png`, `residualplot_hist.png`.
>>>>>>> b146ea7f6f5a2f661933164502f9eab3c9613ad8

## Kjente problemer (må fikses før videre modellering)

1. ~~**SARIMA-evalueringen er ikke fair.**~~ ✅ Løst 2026-04-29 i `sarima_eksperiment.py` via walk-forward med `append(refit=False)`. Tallene i resultattabellen er fra rullende opphav.
2. **FAO-data dropping er en proxy.** Kolonnene fjernes helt fordi 2023+ er `NaN`.
   Forward-fill 2022-verdien + en `fao_imputert`-flagg lar oss beholde signalet
   uten leakage. Beslutning ikke tatt. (Spor B)
3. **Lag-features har NaN i de første ~52 radene.** Notebooken dropper dem implisitt
   via `dropna()` i XGBoost-blokken, men ikke for naiv/SARIMA. Greit nå, men
   sammenligningsgrunnlag (`n` per modell) bør sjekkes ved rapport. SARIMA-rullende
   har n=101/97/93 (per h=4/8/12) fordi siste h-1 uker mangler fasit på t+h.
4. **SARIMA-CI underdekker** (~80 % vs. 95 % nominelt) på grunn av fat-tailed
   residualer. Punktprognoser er OK; intervallene må enten rapporteres med dette
   forbeholdet, eller erstattes med empirisk kalibrerte (kvantil-basert eller
   bootstrap). (Spor A nice-to-have / rapportering)
5. **Restautokorrelasjon ved sesonglag** (Ljung-Box ved lag 52 forkastet for begge
   modeller). Indikerer at (1,1,1)(1,1,1,52)-orden ikke fanger hele
   sesongstrukturen. Auto-ARIMA kan teste høyere orden. (Spor A nice-to-have)

## Neste steg (prioritert)

<<<<<<< HEAD
1. ~~**Fiks SARIMA-evalueringen** med rullende opphav.~~ ✅ Spor A 2026-04-29.
2. **Bestem FAO-håndtering** og kjør XGBoost på nytt med fao-features inkludert. (Spor B)
3. **Hyperparameter-tuning XGBoost** (`RandomizedSearchCV` + `TimeSeriesSplit`)
   – nåværende parametre er ubevisst valgt. (Spor B)
4. ~~**SARIMAX med valuta som eksogen regressor**~~ ✅ Spor A 2026-04-29 — marginal gevinst kun på h=12.
5. **LightGBM** som alternativ ML-modell (nevnt i prosjektplanen). (Spor B)
6. ~~**Konfidensintervaller** for SARIMA-prognoser i sluttrapport.~~ ✅ Spor A 2026-04-29 — lagret i `resultater/sarima_prognose_h*.csv`.
7. **Ekstra feature engineering**: differanser, EUR/USD-ratio, akkumulert volum. (Spor C)
8. **Auto-ARIMA** (`pmdarima`) for å verifisere at (1,1,1)(1,1,1,52) er rimelig orden. (Spor A – nice-to-have)
9. **Sensitivitetsanalyse** av refit-frekvens for SARIMAX (refit=False vs. refit hver 12. uke). (Spor A – nice-to-have)
=======
1. **Fiks SARIMA-evalueringen** med rullende opphav. Dette er forutsetningen for
   at sammenligningen mot XGBoost skal være meningsfull.
2. ~~**Bestem FAO-håndtering**~~ **Ferdig (Spor C, 2026-04-29)** – FAO-kolonnene er forward-filla for 2023–2026; `fao_imputert`-flagg skiller observerte fra interpolerte rader. XGBoost kan no bruke FAO-features utan å droppe dei.
3. ~~**Hyperparameter-tuning XGBoost** (`RandomizedSearchCV` + `TimeSeriesSplit`)~~ **Ferdig (Spor B, 2026-04-29)** – tunet XGBoost og LightGBM, resultat i `resultater/xgboost_tunet.csv` og `lgbm_tunet.csv`.
4. **SARIMAX med valuta som eksogen regressor** (`eur_nok_snitt`, `usd_nok_snitt`).
5. ~~**LightGBM** som alternativ ML-modell (nevnt i prosjektplanen).~~ **Ferdig (Spor B, 2026-04-29)** – LightGBM slår naiv på h=8 og h=12.
6. **Konfidensintervaller** for SARIMA-prognoser i sluttrapport.
7. ~~**Ekstra feature engineering**: differanser, EUR/USD-ratio, akkumulert volum.~~ **Ferdig (Spor C, 2026-04-29)** – 12 nye kolonner tilgjengelig i `laks_ukentlig_features.csv` (se datamappa LES_MEG).
8. ~~**Full retren og residualplot**~~ **Ferdig (Spor B, 2026-04-29)** – alle Spor C-features + FAO inkludert; ensemble slår naiv på alle tre horisontar; residualplot i `resultater/`.
>>>>>>> b146ea7f6f5a2f661933164502f9eab3c9613ad8

## Tilstand i notebooken

Alle celler kjører uten feil per 2026-04-29. Outputs (tabeller, plot, feature
importance) er lagret i `baseline_modeller.ipynb`.
