# 2. Metode

## 2.1 Data

### Prisvariabel

Ukentlig gjennomsnittlig eksportpris for fersk hel norsk laks (NOK/kg) er hentet fra Statistisk sentralbyrå (SSB, tabell 08804). Data dekker perioden januar 2010 til mars 2026 — totalt 845 ukentlige observasjoner, hvorav 741 benyttes som treningssett.

### Eksogene variabler

| Kilde | Variabel | Frekvens | Behandling |
|---|---|---|---|
| Norges Bank | EUR/NOK og USD/NOK spot | Daglig | Ukentlig gjennomsnitt (`eur_nok_snitt`, `usd_nok_snitt`) |
| FAO | Fiskeriprisindeks (akvakultur) | Kvartalsvis | Forward-fill til ukentlig; `fao_imputert`-flagg markerer imputerte verdier |

**Merknad om FAO-imputasjon:** Forward-fill gir identisk verdi for alle ~13 påfølgende uker innenfor hvert kvartal. Dette introduserer kunstig autokorrelasjon i `fao_index_raw`-featuret: ML-modellene «ser» en feature som er konstant i 13 uker og deretter hopper til et nytt nivå. Effekten er at modellene kan lære kvartalsskiftsignalet heller enn den underliggende prisutviklingen. `fao_imputert`-flagget gjør det mulig å kontrollere for dette, men ingen eksplisitt korreksjon er gjort utover flaggingen. SHAP-analysen (seksjon 3.5) bekrefter indirekte at FAO-featuret har lavere viktighet enn lagfeaturer på alle horisonter, noe som tyder på at effekten er begrenset i praksis.

### Håndtering av strukturelle avvik og ekstremverdier

To perioder i datasettet skiller seg markant ut:

**COVID-19 (mars–april 2020):** Eksportprisen falt fra ~68 til ~56 NOK/kg over 6 uker (mars–april 2020) som følge av redusert etterspørsel og logistikkforstyrrelser. Disse observasjonene er beholdt i treningsdataene (prisutslagene er reelle og representative for markedssjokk), men perioden er identifisert og merket i EDA-analysen. COVID-perioden faller utelukkende innenfor treningssettet (test starter uke 12, 2024) og påvirker dermed ikke evalueringen direkte.

**Regimeskift 2022–2023:** Lakseprisen steg fra ~70 NOK/kg (Q3 2022) til over 120 NOK/kg (Q1 2023). Dette representerer en strukturell endring i prisdynamikken og er ikke en klassisk utligger, men snarere et regimeskift. Observasjonene er beholdt uendret; effekten diskuteres inngående i seksjon 4.2.

Ingen uker er ekskludert fra treningssettet basert på regelmessig outlier-fjerning, da fjernelse av ekstreme observasjoner ville gitt et uriktig bilde av den faktiske prisvolatiliteten modellene skal håndtere.

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
- **EUR/NOK og USD/NOK:** `eur_nok_snitt`, `usd_nok_snitt`, og avledede features

De første ~52 radene inneholder NaN-verdier i lag-featurene og er fjernet fra ML-treningssettet via `dropna()`. SARIMA opererer på selve pris-tidsserien og er ikke berørt av denne trimmingen; treningssettene er dermed asymmetriske i startdato, men identiske i sluttdato og testperiode.

## 2.2 Datasplit og evalueringsprotokoll

Datamaterialet deles kronologisk:

- **Treningssett:** alle observasjoner fra 2010 til og med 2021 (siste 104 uker = 2022–2026 holdes ut)
- **Testsett:** siste 104 uker (~2 år)

### Walk-forward-evaluering

For å unngå informasjonslekasje evalueres alle modeller med *walk-forward*-prognose: modellen trenes på alt frem til tidspunkt *t*, lager prognose for *t+4*, *t+8* og *t+12*, deretter forlenges treningsvinduet med én uke og prosessen gjentas. For SARIMA brukes `statsmodels append(refit=False)` — modellparametrene holder seg faste fra initial trening, men modellen oppdateres med hvert nytt datapunkt.

### Evalueringsmetrikker

- **MAE** (Mean Absolute Error, NOK/kg): tolkes direkte som gjennomsnittlig absolutt prisprediksjonsfeil. Primær metrikk.
- **MAPE** (Mean Absolute Percentage Error): muliggjør sammenligning på tvers av prisnivåer.
- **RMSE** (Root Mean Squared Error, NOK/kg): straffer store enkeltfeil hardere enn MAE; brukes som sekundær kontroll.

MAE er valgt som primær rangeringsmetrikk fordi det er robust mot ekstreme enkeltobservasjoner (regimeskift), lineært tolkbart for logistikkplanleggere («gjennomsnittlig avvik er X NOK/kg»), og konsistent med standard prognoseevaluering i råvaremarkeder. Implikasjonene av valget mellom MAE og RMSE drøftes i seksjon 4.6.

## 2.3 Modeller

### 2.3.1 Naiv referansemodell

Prognosegrunnlag: `pris(t + h) = pris(t)`. Tilsvarer ingen modelltilpasning og er minimumsstandarden alle modeller skal slå.

### 2.3.2 SARIMA og SARIMAX

Sesongbasert autoregressiv integrert glidende gjennomsnitt (SARIMA) med orden (1, 1, 1)(1, 1, 1)₅₂, tilpasset med maksimum likelihood-estimering via `statsmodels.tsa.statespace.SARIMAX`. Sesongperiode *m* = 52 reflekterer ukentlig data med et år som naturlig sesong.

**Begrunnelse for ordensvalget:**

Integrasjonsorden *d* = 1 er begrunnet med stasjonaritetstester på treningssettet (n = 741). Augmented Dickey-Fuller-testen (ADF) forkaster *ikke* nullhypotesen om enhetsrot på nivå (t = −1,21, p = 0,668), men forkaster den klart etter første differensiering (t = −7,57, p < 0,001). KPSS-testen bekrefter dette: på nivå forkastes stasjonaritetshypotesen (stat = 0,177, p ≈ 0,025), mens differensert serie ikke forkastes (stat = 0,046, p ≈ 0,10). Seriene er dermed I(1), og *d* = 1 er korrekt.

Sesongdifferensieringsorden *D* = 1 er valgt fordi serien viser et klart, repeterende årssesongmønster (bekreftet av Ljung-Box ved lag 52, p ≪ 0,001 i treningsresidualene). AR- og MA-ordenene *p* = *q* = 1 og *P* = *Q* = 1 følger parsimoniprinsippet: de enkleste ordenene som fanger korttidsdynamikk og sesongstruktur. Modellen oppnår AIC = 3 221,4 og BIC = 3 243,7 på treningssettet.

SARIMAX er identisk bortsett fra at EUR/NOK og USD/NOK-kursene inkluderes som eksogene variabler.

**SARIMAX — oracle vs. naiv valutakursprognose:**

For å gi et rettferdig sammenligningsgrunnlag evalueres SARIMAX i to varianter:

- **SARIMAX_oracle:** De faktiske (realiserte) valutakursene for prognoseperioden brukes som `exog_test`. Dette gir en *optimistisk øvre grense* for hva SARIMAX kan oppnå med perfekt valutainformasjon, men er ikke sammenlignbar med øvrige modeller som ikke har tilgang til fremtidig informasjon.
- **SARIMAX_naiv:** Ved hvert prognosetrinn t benyttes den siste kjente valutakursverdien som prognose for alle fremtidige steg (random walk). Dette er den standardprognosen som er dokumentert å være vanskelig å slå for valutakurser (Meese & Rogoff, 1983), og representerer en rettferdig operativ sammenligning.

SARIMAX_naiv oppnår AIC = 3 222,4 på treningssettet. De to variantenes ytelse sammenlignes direkte i tabell 3.1b.

Konfidensintervaller (95 %) leveres av SARIMAX-objektets `get_forecast()` med Gauss-antagelse.

### 2.3.3 XGBoost og LightGBM med early stopping

Gradient-boosted beslutningstrær tilpasset med early stopping: 20 % av treningssettet (siste ukene) settes av som valideringssett. Early stopping avbryter trening når valideringstapet ikke forbedres på 50 runder (maks 3000 estimatorer). Alle 44 features benyttes.

Hyperparametere ble søkt med `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=5)`, 60 iterasjoner, MAE-scoring. Beste parametere er dokumentert i tabell 2.2 nedenfor.

**Tabell 2.2 – Endelige hyperparametere (RandomizedSearchCV, MAE-scoring)**

| Parameter | XGBoost h=4 | XGBoost h=8 | XGBoost h=12 | LightGBM h=4 | LightGBM h=8 | LightGBM h=12 |
|---|---:|---:|---:|---:|---:|---:|
| `learning_rate` | 0,20 | 0,20 | 0,05 | 0,20 | 0,10 | 0,20 |
| `max_depth` | 3 | 3 | 3 | – | 8 | 10 |
| `n_estimators` (søk) | 800 | 1 000 | 600 | 400 | 1 000 | 1 000 |
| `subsample` | 0,70 | 0,90 | 1,00 | 0,80 | 1,00 | 0,60 |
| `colsample_bytree` | 0,70 | 0,60 | 1,00 | 0,80 | 0,50 | 0,80 |
| `reg_lambda` | 1,0 | 5,0 | 0,5 | 0,5 | 0,5 | 5,0 |
| `reg_alpha` | 0,00 | 0,10 | 0,01 | 0,01 | 0,01 | 0,00 |
| `num_leaves` (LGBM) | – | – | – | 63 | 63 | 127 |
| `min_child_samples` (LGBM) | – | – | – | 20 | 20 | 50 |

Early stopping bestemmer det endelige antall estimatorer ved å monitere valideringstapet, og vil avvike fra verdiene i tabellen ovenfor.

### 2.3.4 Ensemble

Ensemble-prognosen er et vektet snitt av XGBoost+ES og LightGBM+ES:
`y_hat = w · XGB + (1-w) · LGBM`

Lik vekting (*w* = 0,5) benyttes i den primære sammenligningen — dette er det eneste operativt forsvarlige valget, da det ikke krever informasjon fra testsettet. Analyse av post-hoc optimale vekter (basert på kjente testresidualer) er presentert i diskusjonskapittelet (seksjon 4.5) utelukkende som illustrasjon av potensiell gevinst ved fremtidig adaptiv vekting.

## 2.4 Usikkerhetskvantifisering

I tillegg til Gauss-CI fra SARIMA/SARIMAX ble to empiriske metoder undersøkt:

- **Bootstrap:** In-sample-residualer fra SARIMA/SARIMAX skaleres til prognosehorisonten og brukes til å simulere 2 000 fremtidsforløp. 2,5- og 97,5-persentilene gir empirisk CI.
- **Kvantilregresjon (LightGBM):** LightGBM trenes separat for kvantilene 0,025, 0,5 og 0,975 (`objective="quantile"`) for direkte estimering av prediksjonsbånd.

## 2.5 Statistisk signifikanstesting (Diebold-Mariano)

For å avgjøre om observerte ytelsesforskjeller mellom modeller er statistisk meningfulle, benyttes Diebold-Mariano-testen (Diebold & Mariano, 1995). Testen evaluerer nullhypotesen om lik forventet prediksjonsnøyaktighet mellom to modeller. Harvey, Leybourne og Newbold (1997) sin small-sample-korreksjon benyttes for å håndtere skjevheter ved små testsett (n ≈ 92–101 observasjoner). Newey-West HAC-varians med båndbredde h–1 korrigerer for autokorrelasjon i tapsdifferansen, som er særlig viktig ved multi-steg-prognoser. Alle tester er to-sidige med MAE som tapsfunksjon.

## 2.6 Tolkning og forklarbarhet (SHAP)

SHAP TreeExplainer (Lundberg & Lee, 2017) brukes til å kvantifisere featuere sin bidrag til LightGBM sine prediksjoner på testsettet. Verdiene angir gjennomsnittlig absolutt SHAP-verdi per feature og horisont.

## 2.7 Metodisk kvalitet: Validitet og reliabilitet

For å sikre studiens vitenskapelige verdi er det gjort eksplisitte vurderinger av validitet og reliabilitet:

*   **Reliabilitet (pålitelighet):** Prosjektets reliabilitet sikres gjennom en transparent dokumentasjon av alle dataprosesseringssteg og bruk av faste tilfeldige frø (random seeds) i maskinlæringsmodellene. Bruken av walk-forward-evaluering på et identisk testsett for alle modeller muliggjør en direkte og rettferdig sammenligning av resultatene.
*   **Intern validitet:** Studien adresserer ikke-stasjonaritet i prisserien gjennom differensiering (i SARIMA) og bruk av lag-features. Den største utfordringen for validiteten er regimeskiftet i 2022–2023, som gjør at modeller trent på historiske data kan ha redusert gyldighet i ekstraordinære perioder. Dette drøftes inngående i kapittel 4.
*   **Ekstern validitet (generaliserbarhet):** Selv om modellene er trent spesifikt på norsk laks, er metodikken (kombinasjon av statistiske modeller og ensemble-ML) overførbar til andre biologiske råvaremarkeder med sesongsvingninger. Bruken av SSB-data og valutakurser sikrer at datagrunnlaget er representativt for det faktiske markedet eksportørene opererer i.
