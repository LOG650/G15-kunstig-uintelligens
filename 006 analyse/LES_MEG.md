# Analyse – baseline-modeller og videre arbeid

Status og overlevering for prediksjon av eksportpris fersk norsk laks.

**Sist oppdatert:** 2026-04-29 (Spor A, B og C ferdig – samlet etter merge)

## Filer i denne mappa

| Fil | Innhold |
|---|---|
| `baseline_modeller.ipynb` | Hovednotebook: laster data, trener Naiv/SARIMA/XGBoost, plotter. **Frosset som referanse – ikke endre** |
| `eksporter_baseline.py` | Headless-versjon av samme logikk; skriver metrikker til `resultater/`. **Frosset – ikke endre** |
| `_bygg_notebook.py` | Hjelpeskript brukt til å generere notebook-skjelettet |
| `oppgaver/spor_*.md` | Selvstendige prompts per arbeidsspor – lim inn i AI-en din |
| `sarima_eksperiment.py` | **Spor A** – SARIMA / SARIMAX med rullende opphav, 95 % CI, og residualdiagnostikk |
| `ml_eksperiment.py` | **Spor B** – FAO-sammenligning, XGBoost/LightGBM-tuning, feature importance |
| `ml_ensemble.py` | **Spor B** – Ensemble XGBoost + LightGBM med early stopping; lagrer `ml_ensemble_prediksjoner.csv` |
| `ml_residualplot.py` | **Spor B** – Residualanalyse av ensemble; produserer 3 plotfiler |
| `resultater/baseline_metrikker*.csv` | Naiv/SARIMA/XGBoost utuned (fra baseline-notebooken) |
| `resultater/sarima_metrikker.csv` | MAE/MAPE for SARIMA og SARIMAX (rullende opphav) |
| `resultater/sarima_prognose_h{4,8,12}.csv`, `sarimax_prognose_h{4,8,12}.csv` | Punktprognoser med 95 % CI |
| `resultater/sarima_ci_dekning.csv` | Empirisk dekning av 95 %-CI (mål: 0.95) |
| `resultater/sarima_residualdiagnostikk.csv` | Ljung-Box + Jarque-Bera på treningsresidualer |
| `resultater/sarima_prognoser.png`, `sarima_residualer.png`, `sarimax_residualer.png` | Plot fra Spor A |
| `resultater/xgboost_tunet.csv`, `lgbm_tunet.csv` | Tunede ML-modeller (60 iter RandomizedSearchCV) |
| `resultater/ml_fao_sammenligning.csv` | FAO med vs. uten – beslutningsgrunnlag |
| `resultater/ml_ensemble.csv`, `ml_ensemble_prediksjoner.csv`, `ml_residualar.csv` | Ensemble-prediksjoner og residualer |
| `resultater/ml_feature_importance_h{4,8,12}.{csv,png}` | Top-features per horisont |
| `resultater/ml_mae_samanlikning.png`, `ml_ensemble_prediksjon.png` | Plot fra Spor B |
| `resultater/residualplot_{tid,scatter,hist}.png` | Residualanalyse av ensemble per horisont |

## Parallell arbeidsfordeling – alle spor ferdig

| Spor | Tema | Prompt | Status |
|---|---|---|---|
| **A** | SARIMA / SARIMAX / KI / residualdiagnostikk | [`oppgaver/spor_a_sarima.md`](oppgaver/spor_a_sarima.md) | ✅ Ferdig |
| **B** | XGBoost / LightGBM / tuning / ensemble | [`oppgaver/spor_b_ml.md`](oppgaver/spor_b_ml.md) | ✅ Ferdig |
| **C** | Feature engineering + FAO-imputation | [`oppgaver/spor_c_features.md`](oppgaver/spor_c_features.md) | ✅ Ferdig |

Konfliktregler (hold ved videre arbeid):
- `baseline_modeller.ipynb` og `eksporter_baseline.py` er FROSSET. Ingen endrer disse.
- Spor C kan kun **legge til** kolonner i features-CSV, aldri rename eller slette.
- Alle skriver resultater til `resultater/<eget-prefix>_*.csv`. Aldri overskriv andres filer.

## Reproduksjon

```bash
cd "006 analyse"

# Baseline (frosset referanse)
python eksporter_baseline.py

# Spor A: SARIMA / SARIMAX rullende opphav + diagnostikk
python sarima_eksperiment.py

# Spor B: FAO-sammenligning, tuning, feature importance
python ml_eksperiment.py
# Spor B: ensemble med early stopping (lagrer ml_ensemble_prediksjoner.csv)
python ml_ensemble.py
# Spor B: residualplot (forutsetter ml_ensemble_prediksjoner.csv)
python ml_residualplot.py
```

Krever `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `xgboost`, `lightgbm`, `matplotlib`, `scipy`.
Datasettet leses fra `../004 data/Analyseklart datasett/laks_ukentlig_features.csv` (44 kolonner inkl. Spor C-features og FAO-imputation).

## Oppsett

- **Testset:** siste 104 uker (~2 år), kronologisk split
- **Horisonter:** 4, 8, 12 uker
- **Metrikker:** MAE, MAPE
- **FAO-håndtering:** Spor C forward-filte FAO-verdiene fra 2022 inn i 2023–2026 og la til en `fao_imputert`-flagg. Spor B sin sammenligning viste at FAO-features med tunet XGBoost gir ~0.3 NOK/kg lavere MAE på h=4 og h=8, men dårligere på h=12. Ensemblet bruker FAO via early stopping som demper overfitting.

## Samlet resultattabell – test (siste 104 uker)

MAE i NOK/kg på testperioden, beste per horisont i fet skrift.

| Modell | h=4 MAE | h=4 MAPE | h=8 MAE | h=8 MAPE | h=12 MAE | h=12 MAPE |
|---|---:|---:|---:|---:|---:|---:|
| Naiv (`pris(t-h)`) | 8.51 | 9.8 % | 13.04 | 15.4 % | 16.35 | 19.7 % |
| **SARIMA** rullende¹ | **8.27** | **9.5 %** | 11.07 | 13.1 % | 13.15 | 15.9 % |
| **SARIMAX** rullende¹ (EUR/USD) | 8.33 | 9.6 % | 11.07 | 13.1 % | **12.93** | **15.6 %** |
| XGBoost utuned (baseline) | 11.46 | 13.7 % | 12.59 | 15.1 % | 14.79 | 18.1 % |
| XGBoost tunet² | 10.37 | 12.2 % | 11.98 | 14.3 % | 15.47 | 19.0 % |
| LightGBM tunet² | 10.45 | 12.3 % | 10.90 | 12.9 % | 13.06 | 16.0 % |
| XGBoost + early stopping³ | 8.71 | 10.1 % | 10.88 | 12.9 % | 15.31 | 18.7 % |
| LightGBM + early stopping³ | 8.85 | 10.4 % | 11.53 | 13.7 % | 13.24 | 16.3 % |
| **Ensemble** (XGB+ES + LGBM+ES, snitt)³ | 8.33 | 9.6 % | **10.85** | **12.9 %** | 13.56 | 16.7 % |

¹ Rullende-opphav-evaluering (`append(refit=False)` per uke). 36 trening + 105 test obs SARIMA, samme + EUR/USD-eksogene SARIMAX. AIC ≈ 3222 begge.
² `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=5)`, 60 iter, MAE-scoring. 36 features (uten FAO; FAO+uten-FAO gav uavgjort i snitt 12.183 vs 12.184).
³ Early stopping: 40 features inkl. FAO + `fao_imputert`-flagg. ES stoppet etter 17–37 iterasjoner.

**Hovedfunn:**
- **SARIMA er beste enkeltmodell på h=4** (8.27), tett etterfulgt av SARIMAX og Ensemble (begge 8.33).
- **Ensemble vinner h=8** (10.85), så vidt foran XGBoost+ES (10.88) og LightGBM tunet (10.90).
- **SARIMAX vinner h=12** (12.93), mens LightGBM tunet (13.06) og LightGBM+ES (13.24) er nest best – ensemble-averaging skader på lang horisont fordi XGBoost+ES er svakere her.
- Alle de øverste modellene slår naiv på alle horisonter; ren XGBoost-tuning uten early stopping er fortsatt svakere enn naiv på h=4.

### CI-kalibrering og residualdiagnostikk (SARIMA / SARIMAX)

Empirisk dekning av 95 %-konfidensintervallene:

| Modell | h=4 | h=8 | h=12 |
|---|---:|---:|---:|
| SARIMA | 79 % | 80 % | 81 % |
| SARIMAX | 81 % | 81 % | 82 % |

CI-ene er **systematisk for smale** (~80 % faktisk dekning vs. 95 % nominell). Dette er konsistent med residualdiagnostikken (standardiserte treningsresidualer):

| Modell | LB(10) p | LB(20) p | LB(52) p | JB skew | JB kurtose |
|---|---:|---:|---:|---:|---:|
| SARIMA  | 0.004 | 0.002 | <0.001 | +0.24 | 4.55 |
| SARIMAX | 0.009 | 0.004 | <0.001 | +0.28 | 4.44 |

- **Ljung-Box forkaster hvit-støy-hypotesen på alle lag** – modellen fanger ikke all autokorrelasjon, særlig sesongleddet ved lag 52.
- **Residualene har fetere haler enn normal** (kurtose ≈ 4.5 vs. 3) – derfor er Gauss-baserte CI-er for trange.
- For rapporten: punktprognoser er rimelige, men intervallene gir falsk presisjon.

### Residualanalyse – ensemble (Spor B)

| Horisont | n | MAE | Bias | Std |
|---:|---:|---:|---:|---:|
| 4 | 100 | 8.33 | -2.16 | 10.56 |
| 8 | 96 | 10.85 | -2.92 | 13.44 |
| 12 | 92 | 13.56 | -2.73 | 17.03 |

**Systematisk negativ bias:** ensemblet underpredikerer konsekvent med ~2.2–2.9 NOK/kg på tvers av alle horisonter. Sett `residualplot_tid.png`, `residualplot_scatter.png`, `residualplot_hist.png` for visuelle diagnoser.

## Kjente problemer / forbehold

1. ~~SARIMA-evalueringen er ikke fair~~ ✅ Løst (Spor A) – walk-forward med `append(refit=False)`.
2. ~~FAO-data dropping er en proxy~~ ✅ Løst (Spor C) – forward-fill + `fao_imputert`-flagg. Spor B viste at FAO med early stopping inkluderes uten å skade.
3. **Lag-features har NaN i de første ~52 radene.** Sammenligningsgrunnlag (`n` per modell) bør sjekkes ved rapport. Aktuelle n: SARIMA/SARIMAX 101/97/93, ML 100/96/92.
4. **SARIMA-CI underdekker** (~80 % vs. 95 %) på grunn av fat-tailed residualer. Bruk forbehold i rapport, eller bytt til empirisk kalibrerte intervaller.
5. **Restautokorrelasjon ved sesonglag 52** (Ljung-Box forkastet for begge SARIMA-varianter). Indikerer at (1,1,1)(1,1,1,52)-orden ikke fanger hele sesongstrukturen.
6. **Ensemble har systematisk negativ bias** (-2.2 til -2.9 NOK/kg). Bør undersøkes i rapport: bias-korreksjon? Asymmetrisk loss?

## Neste steg (gjenstående, prioritert)

1. ~~Fiks SARIMA-evaluering~~ ✅ Spor A
2. ~~FAO-håndtering~~ ✅ Spor C
3. ~~Hyperparameter-tuning XGBoost~~ ✅ Spor B
4. ~~SARIMAX med valuta som eksogen~~ ✅ Spor A
5. ~~LightGBM som alternativ~~ ✅ Spor B
6. ~~Konfidensintervaller for SARIMA~~ ✅ Spor A
7. ~~Ekstra feature engineering~~ ✅ Spor C (12 nye kolonner)
8. ~~Full retren med Spor C-features + ensemble~~ ✅ Spor B

**Gjenstående (nice-to-have / rapportering):**

a. **Rapport-skriving** – tabeller, plot og diskusjon av modellbegrensningene (CI-underdekning, bias, sesongautokorrelasjon).
b. **Auto-ARIMA** (`pmdarima`) for å verifisere at (1,1,1)(1,1,1,52) er rimelig orden – ev. teste høyere sesongorden gitt LB-funn.
c. **Sensitivitet på refit-frekvens** for SARIMAX (refit=False vs. refit hver 12. uke) – validerer Spor A's hurtigløsning.
d. **Bias-korreksjon på ensemble** – enkel post-hoc-justering basert på treningsresidualer.
e. **Empirisk kalibrerte intervaller** (kvantilregresjon eller bootstrap) som erstatning for de underdekkende Gauss-CI-ene.
f. **Ensemble-vekting** – ulik vekt på XGBoost og LightGBM per horisont (LightGBM dominerer h=12).

## Tilstand i notebooken

`baseline_modeller.ipynb` er frosset som referanse fra 2026-04-29; alle celler kjører uten feil. Outputs i notebooken kan være utdaterte sammenlignet med `resultater/` og denne LES_MEG; bruk denne fila som sannhet.
