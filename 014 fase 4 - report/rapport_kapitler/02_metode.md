# 2. Metode

## 2.1 Data

### Prisvariabel

Ukentlig gjennomsnittlig eksportpris for fersk hel norsk laks (NOK/kg) er hentet fra Statistisk sentralbyrå (SSB, tabell 08804). Data dekker perioden januar 2009 til mars 2024 — totalt ~789 ukentlige observasjoner.

### Eksogene variabler

| Kilde | Variabel | Frekvens | Behandling |
|---|---|---|---|
| Norges Bank | EUR/USD spot | Daglig | Ukentlig gjennomsnitt |
| FAO | Fiskeriprisindeks (akvakultur) | Kvartalsvis | Forward-fill til ukentlig; `fao_imputert`-flagg markerer imputerte verdier |

### Tidsseriekarakteristika og deskriptiv statistikk

Tabell 2.1 oppsummerer de viktigste statistiske egenskapene til eksportprisserien over hele analyseperioden (n = 743 ukentlige observasjoner, 2010–2024).

**Tabell 2.1 – Deskriptiv statistikk for eksportpris (NOK/kg)**

| Statistikk | Verdi |
|---|---:|
| Gjennomsnitt | 55,1 NOK/kg |
| Median | 53,9 NOK/kg |
| Standardavvik | 20,5 NOK/kg |
| Minimum | 21,6 NOK/kg (uke 43, 2011) |
| Maksimum | 122,9 NOK/kg (uke 11, 2023) |
| Ukentlig prisendring, std | 2,93 NOK/kg |

Variasjonskoeffisienten (~37 %) bekrefter at lakseprisen er svært volatil sammenlignet med de fleste industrielle råvarer. Ukentlige enkeltbevegelser på ±12 NOK/kg er observert.

**Sesongmønster:** Gjennomsnittsprisen varierer systematisk gjennom året: Q1 og Q2 (vinter/vår) er høyest (~59 og 58 NOK/kg), mens Q3 (sommer) er lavest (~51 NOK/kg). Dette reflekterer biologisk sesong i oppdrettssyklusen og lavere etterspørsel i sommermånedene. Mønsteret er stabilt over hele perioden, men overlagres av kraftige regimeskift — særlig prisoppgangen 2022–2023 der prisen steg fra ~70 NOK/kg til over 120 NOK/kg på under to år.

**Stasjonaritet:** ADF- og KPSS-tester (se seksjon 2.3.2) bekrefter at serien er I(1): ikke-stasjonær på nivå, men stasjonær etter én differensiering. Dette motiverer bruken av d = 1 i SARIMA-modellen og bruken av lagverdier (i stedet for prisnivå) som features i ML-modellene.

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

**Begrunnelse for ordensvalget:**

Integrasjonsorden *d* = 1 er begrunnet med stasjonaritetstester på treningssettet (n = 741). Augmented Dickey-Fuller-testen (ADF) forkaster *ikke* nullhypotesen om enhetsrot på nivå (t = −1,21, p = 0,668), men forkaster den klart etter første differensiering (t = −7,57, p < 0,001). KPSS-testen bekrefter dette: på nivå forkastes stasjonaritetshypotesen (stat = 0,177, p ≈ 0,025), mens differensert serie ikke forkastes (stat = 0,046, p ≈ 0,10). Seriene er dermed I(1), og *d* = 1 er korrekt.

Sesongdifferensieringsorden *D* = 1 er valgt fordi serien viser et klart, repeterende årssesongmønster (bekreftet av Ljung-Box ved lag 52, p ≪ 0,001 i treningsresidualene). AR- og MA-ordenene *p* = *q* = 1 og *P* = *Q* = 1 følger parsimoniprinsippet: de enkleste ordenene som fanger korttidsdynamikk og sesongstruktur. Modellen oppnår AIC = 3 221,4 og BIC = 3 243,7 på treningssettet.

SARIMAX er identisk bortsett fra at EUR/USD-kursen inkluderes som eksogen variabel — tilpasset og prognositisert parallelt med prisvariabelen.

I walk-forward-evalueringen benyttes de faktiske (realiserte) EUR/USD-verdiene for prognoseperioden (`exog_test`), ikke en fremoverrettet valutaprognose. Dette gir en optimistisk øvre grense for hva SARIMAX kan oppnå med perfekt valutainformasjon; i operativt bruk ville EUR/USD-kursen for de kommende 4–12 ukene måtte prognoseres separat, noe som vil introdusere ytterligere usikkerhet.

Konfidensintervaller (95 %) leveres av SARIMAX-objektets `get_forecast()` med Gauss-antagelse.

### 2.3.3 XGBoost og LightGBM med early stopping

Gradient-boosted beslutningstrær tilpasset med early stopping: 20 % av treningssettet (siste ukene) settes av som valideringssett. Early stopping avbryter trening når valideringstapet ikke forbedres på 50 runder (maks 3000 estimatorer). Alle 44 features benyttes.

Hyperparametere ble søkt med `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=5)`, 60 iterasjoner, MAE-scoring. Beste parametere lagres i `resultater/xgboost_tunet.csv` og `lgbm_tunet.csv`.

### 2.3.4 Ensemble

Ensemble-prognosen er et vektet snitt av XGBoost+ES og LightGBM+ES:
`y_hat = w · XGB + (1-w) · LGBM`

Lik vekting (*w* = 0,5) brukes for h = 4. For h = 8 og h = 12 er optimale vekter identifisert post-hoc via grid search over *w* ∈ {0,0; 0,1; …; 1,0} (se seksjon 3.4).

## 2.4 Usikkerhetskvantifisering

I tillegg til Gauss-CI fra SARIMA/SARIMAX ble to empiriske metoder undersøkt:

- **Bootstrap:** In-sample-residualer fra SARIMA/SARIMAX skaleres til prognosehorisonten og brukes til å simulere 2 000 fremtidsforløp. 2,5- og 97,5-persentilene gir empirisk CI.
- **Kvantilregresjon (LightGBM):** LightGBM trenes separat for kvantilene 0,025, 0,5 og 0,975 (`objective="quantile"`) for direkte estimering av prediksjonsbånd.

## 2.5 Tolkning og forklarbarhet (SHAP)

SHAP TreeExplainer (Lundberg & Lee, 2017) brukes til å kvantifisere featuere sin bidrag til LightGBM sine prediksjoner på testsettet. Verdiene angir gjennomsnittlig absolutt SHAP-verdi per feature og horisont.
