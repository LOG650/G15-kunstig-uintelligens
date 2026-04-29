# Spor B – ML-sporet (XGBoost / LightGBM)

Lim hele dette dokumentet inn som første melding til AI-en. Den får da nok kontekst til å jobbe selvstendig.

---

## Prosjektkontekst

Studentprosjekt LOG650 (Logistikk og kunstig intelligens) ved HiM. Vi skal predikere ukentlig eksportpris for fersk norsk laks (NOK/kg) på horisontene 4, 8 og 12 uker fram. Tre studenter jobber parallelt; du jobber kun på ML-sporet (gradient boosting).

Arbeidskatalog: `006 analyse/`. Datasettet ligger i `004 data/Analyseklart datasett/laks_ukentlig_features.csv` og er beskrevet i `004 data/Analyseklart datasett/LES_MEG.md`. Periode: 2010-01-04 til 2026-03-09 (845 ukentlige observasjoner). Testset = siste 104 uker. Mål: MAE og MAPE per horisont.

## Baseline du må slå (fra `006 analyse/resultater/baseline_metrikker_pivot.csv`)

| Horisont | Naiv MAE | XGBoost MAE (utuned) |
|---|---|---|
| 4 | **8.51** | 11.46 |
| 8 | **13.04** | 12.59 |
| 12 | 16.35 | **14.79** |

Naiv slår XGBoost på h=4 og h=8. Det skal du fikse.

## Dine oppgaver (i prioritert rekkefølge)

1. **FAO-håndtering.** I baseline droppes `fao_global_atlantisk_tonn`, `fao_norge_tonn`, `fao_eks_norge_tonn` fordi de er `NaN` fra 2023. Test forward-fill av siste 2022-verdi + en binær `fao_imputert`-flagg som markerer rader hvor FAO er ekstrapolert. Sammenlign mot baseline-XGBoost (uten FAO). Behold den varianten som gir best testset-MAE og bruk den videre.
2. **Hyperparameter-tuning XGBoost** med `RandomizedSearchCV` + `TimeSeriesSplit` (n_splits=5). Søkerom: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`. Bruk MAE som scoring. Egen modell per horisont (h=4, 8, 12).
3. **LightGBM som alternativ.** Tren tilsvarende `LGBMRegressor` med samme features og samme tuning-protokoll. Sammenlign mot tunet XGBoost.
4. **Feature importance** for vinnermodellen på hver horisont (top 15).

Bruk samme tidsbaserte split som baseline: cutoff på `train.index.max()` (= 104 uker før slutt). Inni TimeSeriesSplit må folds være kronologiske.

## Filer du eier (lag/endre fritt)

- `006 analyse/ml_eksperiment.py` – hovedskript (kan opprettes)
- `006 analyse/ml_arbeid.ipynb` – eksplorativ notebook hvis du vil (kan opprettes)
- `006 analyse/resultater/xgboost_tunet.csv` – output med kolonner `horisont, n, MAE, MAPE, beste_params`
- `006 analyse/resultater/lgbm_tunet.csv` – tilsvarende for LightGBM
- `006 analyse/resultater/ml_feature_importance_h{4,8,12}.csv`
- `006 analyse/resultater/ml_*.png` – figurer

## Filer du IKKE skal røre

- `006 analyse/baseline_modeller.ipynb` – frosset som referanse
- `006 analyse/eksporter_baseline.py` – frosset
- `006 analyse/sarima_*.py`, `sarima_*.ipynb` – tilhører Spor A
- `004 data/` og alle filer der – tilhører Spor C
- Du kan LESE features-CSV-en, men IKKE bygg datasettet på nytt

## Hvis Spor C legger til nye features

Spor C kan legge til nye kolonner i `laks_ukentlig_features.csv` mens du jobber. De skal aldri rename eller fjerne eksisterende kolonner. Når du `git pull` og ser nye kolonner: vurder om de skal med i feature-listen din. Standardregelen er ja, så lenge de ikke lekker fremtidig info (sjekk at lag/shift er gjort).

## Slik oppdaterer du delt status

Når du er ferdig med en oppgave, oppdatér `006 analyse/LES_MEG.md`:
- Bytt ut XGBoost-radene i resultattabellen med tunede tall
- Legg til en LightGBM-rad
- Stryk relevante punkt under "Neste steg" (FAO-håndtering, XGBoost-tuning, LightGBM)
- Oppdatér "Sist oppdatert"-datoen

Hvis du får merge-konflikt på LES_MEG.md, behold begge endringene – tabellen og listene er additive.

## Suksesskriterium

Den beste tunede ML-modellen (XGBoost eller LightGBM) slår naiv baseline på alle tre horisontene målt på MAE.
