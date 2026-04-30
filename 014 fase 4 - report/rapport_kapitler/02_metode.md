# 2. Metode

## 2.1 Data

### Prisvariabel

Ukentlig gjennomsnittlig eksportpris for fersk hel norsk laks (NOK/kg) er hentet fra Statistisk sentralbyrå (SSB, tabell 08804). Data dekker perioden januar 2009 til mars 2024 — totalt ~789 ukentlige observasjoner.

### Eksogene variabler

| Kilde | Variabel | Frekvens | Behandling |
|---|---|---|---|
| Norges Bank | EUR/USD spot | Daglig | Ukentlig gjennomsnitt |
| FAO | Fiskeriprisindeks (akvakultur) | Kvartalsvis | Forward-fill til ukentlig; `fao_imputert`-flagg markerer imputerte verdier |

### Feature-engineering

Følgende feature-grupper ble konstruert fra prisvariabelen og eksogene variabler (44 kolonner totalt):

- **Lagverdier:** `pris_lag_1` til `pris_lag_52`
- **Glidende gjennomsnitt:** `pris_ma_4`, `pris_ma_8`, `pris_ma_12`, `pris_ma_26`, `pris_ma_52`
- **Volatilitet:** `pris_std_4`, `pris_std_8`, `pris_std_12`
- **Sesong:** `uke_sin`, `uke_cos` (sirkulær koding av ukenummer 1–52)
- **Eksportvolum:** `volum_sum_4u`, `volum_sum_12u`, `volum_sum_52u` (SSB)
- **FAO-index:** `fao_index_raw`, `fao_imputert`
- **EUR/USD:** `eur_usd`, `eur_usd_lag_1`, `eur_usd_lag_4`

De første ~52 radene inneholder NaN-verdier i lag-featurene og er fjernet fra ML-treningssettet via `dropna()`.

## 2.2 Datasplit og evalueringsprotokoll

Datamaterialet deles kronologisk:

- **Treningssett:** alle observasjoner fra 2009 til og med 2021 (siste 104 uker = 2022–2024 holdes ut)
- **Testsett:** siste 104 uker (~2 år)

### Walk-forward-evaluering

For å unngå informasjonslekasje evalueres alle modeller med *walk-forward*-prognose: modellen trenes på alt frem til tidspunkt *t*, lager prognose for *t+4*, *t+8* og *t+12*, deretter forlenges treningsvinduet med én uke og prosessen gjentas. For SARIMA brukes `statsmodels append(refit=False)` — modellparametrene holder seg faste fra initial trening, men modellen oppdateres med hvert nytt datapunkt.

### Evalueringsmetrikker

- **MAE** (Mean Absolute Error, NOK/kg): tolkes direkte som gjennomsnittlig absolutt prisprediksjonsfeil.
- **MAPE** (Mean Absolute Percentage Error): muliggjør sammenligning på tvers av prisnivåer.

## 2.3 Modeller

### 2.3.1 Naiv referansemodell

Prognosegrunnlag: `pris(t + h) = pris(t)`. Tilsvarer ingen modelltilpasning og er minimumsstandarden alle modeller skal slå.

### 2.3.2 SARIMA og SARIMAX

Sesongbasert autoregressiv integrert glidende gjennomsnitt (SARIMA) med orden (1, 1, 1)(1, 1, 1)₅₂, tilpasset med maksimum likelihood-estimering via `statsmodels.tsa.statespace.SARIMAX`. Sesongperiode *m* = 52 reflekterer ukentlig data med et år som naturlig sesong.

SARIMAX er identisk bortsett fra at EUR/USD-kursen inkluderes som eksogen variabel — tilpasset og prognositisert parallelt med prisvariabelen.

Konfidensintervaller (95 %) leveres av SARIMAX-objektets `get_forecast()` med Gauss-antagelse.

### 2.3.3 XGBoost og LightGBM med early stopping

Gradient-boosted beslutningstrær tilpasset med early stopping: 20 % av treningssettet (siste ukene) settes av som valideringssett. Early stopping avbryter trening når valideringstapet ikke forbedres på 50 runder (maks 3000 estimatorer). Alle 44 features benyttes.

Hyperparametere ble søkt med `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=5)`, 60 iterasjoner, MAE-scoring. Beste parametere lagres i `resultater/xgboost_tunet.csv` og `lgbm_tunet.csv`.

### 2.3.4 Ensemble

Ensemble-prognosen er et vektet snitt av XGBoost+ES og LightGBM+ES:
`y_hat = w · XGB + (1-w) · LGBM`

Lik vekting (*w* = 0,5) brukes for h = 4. For h = 8 og h = 12 er optimale vekter identifisert post-hoc via grid search over *w* ∈ {0,0; 0,1; …; 1,0} (se seksjon 3.4).

## 2.4 Usikkerhetskvantifisering

I tillegg til Gauss-CI fra SARIMA/SARIMAX ble to empiriske metoder undersøkt (Spor G):

- **Bootstrap:** In-sample-residualer fra SARIMA/SARIMAX skaleres til prognosehorisonten og brukes til å simulere 2 000 fremtidsforløp. 2,5- og 97,5-persentilene gir empirisk CI.
- **Kvantilregresjon (LightGBM):** LightGBM trenes separat for kvantilene 0,025, 0,5 og 0,975 (`objective="quantile"`) for direkte estimering av prediksjonsbånd.

## 2.5 Tolkning og forklarbarhet (SHAP)

SHAP TreeExplainer (Lundberg & Lee, 2017) brukes til å kvantifisere featuere sin bidrag til LightGBM sine prediksjoner på testsettet. Verdiene angir gjennomsnittlig absolutt SHAP-verdi per feature og horisont.
