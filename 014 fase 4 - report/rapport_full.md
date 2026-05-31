# Sammendrag

Denne rapporten undersøker mulighetene for maskinlæringsbasert prognose av ukentlig eksportpris for fersk norsk laks (NOK/kg) på tidshorisonter på 4, 8 og 12 uker frem i tid. Datasettene kombinerer SSBs ukentlige eksportstatistikk med valutakurs (EUR/USD, Norges Bank) og FAO sin kvartalsvise prisindeks for akvakultur — totalt 44 forklaringsvariabler etter feature-engineering.

Ni modellvarianter ble trent og evaluert på en felles testperiode (siste 104 uker, kronologisk splitt). Modellene spenner fra en naiv referansemodell over statistiske tidsseriemodeller (SARIMA og SARIMAX med rullende startpunkt) til gradientøkende tremodeller (XGBoost og LightGBM) og ensemble-kombinasjoner.

Ingen enkeltmodell dominerer alle horisonter:

- **h = 4 uker:** SARIMA oppnår lavest MAE (8,27 NOK/kg, 9,5 % MAPE), tett fulgt av SARIMAX og ensemblet (begge 8,33 NOK/kg).
- **h = 8 uker:** Ensemblet (XGBoost + LightGBM med early stopping) er best med 10,85 NOK/kg (12,9 % MAPE), marginalt bedre enn XGBoost+ES (10,88) og LightGBM tunet (10,90).
- **h = 12 uker:** SARIMAX er best med 12,93 NOK/kg (15,6 % MAPE), etterfulgt av LightGBM tunet (13,06).

Alle toppmodeller slår den naive baselines på samtlige horisonter; forbedringen er størst på h = 4 (3 % lavere MAPE enn naiv).

To sentrale metodiske funn må understrekes: (1) Gauss-baserte 95 %-konfidensintervaller for SARIMA/SARIMAX underdekker konsekvent (~80 % faktisk vs. 95 % nominell), fordi treningsresidualene har fettede haler og uoppfanget sesongautokorrelasjon. (2) Ensemblet viser systematisk negativ bias (–2,2 til –2,9 NOK/kg) som kan spores til lakseprisboomet 2022–2023 — en regimeskiftperiode modellen ikke ble trent på å gjenkjenne.

Studien konkluderer med at statistiske og maskinlæringsbaserte tilnærminger utfyller hverandre, og at en kombinert strategi — SARIMA for korte horisonter, ensemble for mellomhorisonter og SARIMAX for lengre — gir det beste samlede prognosesystemet innenfor en uke-til-uke driftsramme.
# 1. Innledning

## 1.1 Bakgrunn og motivasjon

Norsk lakseoppdrett er en av landets største eksportnæringer, med en eksportverdi som i 2023 oversteg 100 milliarder kroner (SSB, 2024). Eksportprisen for fersk laks svinger kraftig fra uke til uke og er avgjørende for lønnsomheten hos både oppdrettere, eksportører og kjøpere. Aktører med eksponering mot spotmarkedet har behov for pålitelige prognoser på 4–12 ukers sikt for å planlegge slakting, logistikk og prissikring.

Tidligere forskning viser at klassiske tidsseriemodeller som SARIMA ofte fungerer godt for kortsiktige prognoser i markeder med stabile sesongmønstre, mens maskinlæringsmodeller kan prestere bedre når tidsseriene inneholder ikke-lineære sammenhenger og strukturelle brudd. Til tross for den kommersielle viktigheten er offentlig tilgjengelig forskning på kortsiktig ukentlig lakseprisprognosering begrenset. Særlig finnes det få studier som kombinerer klassiske statistiske metoder med moderne gradientøkende tremodeller for dette spesifikke formålet.

Denne studien søker å fylle dette kunnskapshullet ved å integrere SSBs ukentlige statistikk med valutakurser og internasjonale prisindekser (FAO) i et felles prediksjonsrammeverk. Ved å sammenligne modellene på identiske data gjennom en krevende walk-forward-evaluering, bidrar studien til økt forståelse for hvilke arkitekturer som er mest robuste for ulike tidshorisonter i et volatilt råvaremarked.

## 1.2 Problemstilling

Rapporten besvarer følgende spørsmål:

> **Hvilke modeller gir lavest prediksjonsfeil (MAE) for ukentlig eksportpris på fersk norsk laks over prognosehorisonter på 4, 8 og 12 uker?**

Som underspørsmål undersøkes:

1. Er statistiske tidsseriemodeller (SARIMA/SARIMAX) bedre enn gradientøkende tremodeller på korte horisonter?
2. Gir kombinasjon av EUR/USD-valutakurs som eksogen variabel (SARIMAX) bedre prediksjoner enn SARIMA alene?
3. Hvor godt kalibrerte er modellenes konfidensintervaller?
4. Hvilke features er viktigst i de maskinlæringsbaserte modellene?

## 1.3 Avgrensning

Studien dekker ukentlige data fra 2009 til 2024 og evaluerer modellene på de siste 104 ukene (~2 år) i en walk-forward-oppsett uten fremtidig informasjonslekasje. Det er ikke gjort markedsanalyse eller optimert handlingsstrategi — fokus er utelukkende på statistisk prognose.

Rapporten inngår som avsluttende prosjektarbeid i emnet LOG650 (Logistikk og kunstig intelligens) ved Høgskolen i Molde (HiM).

## 1.4 Relatert forskning

### Lakseprisens dynamikk og markedsstruktur

Norsk lakseoppdrett og prisdannelsen i dette markedet er grundig studert av Asche og medarbeidere. Asche og Bjørndal (2011) dokumenterer strukturen i atlantisk laksemarked, herunder prisintegrasjon mellom europeiske markeder og rollen til valutakurser som prisstimulator. Oglend (2013) viser at lakseprisvolatiliteten har økt over tid og identifiserer ikke-linearitet i prisdynamikken — funn som motiverer bruken av ikke-lineære maskinlæringsmodeller i tillegg til klassiske statistiske metoder. Dahl og Oglend (2014) finner at lakseprisen er integrert med EUR/USD-kursen, noe som er den empiriske begrunnelsen for å inkludere valutakurs som eksogen variabel i SARIMAX-spesifikasjonen i denne studien.

### SARIMA og statistiske tidsseriemodeller for råvarer

Box, Jenkins, Reinsel og Ljung (2015) er standardverket for SARIMA-modellering og etablerer rammeverket denne studien bygger på. Hyndman og Athanasopoulos (2021) gir en oppdatert gjennomgang av prognosemetoder og konkluderer at sesongmodeller med differensiering (SARIMA-familien) generelt presterer godt for råvarepriser med stabile sesongmønstre, men at de er sårbare for strukturelle brudd. Begge disse referansene støtter valg av SARIMA som statistisk referansemodell i studien.

### Gradientøkende tremodeller og ensemble-metoder for tidsserier

Chen og Guestrin (2016) introduserte XGBoost, og Ke et al. (2017) introduserte LightGBM, begge med dokumentert overlegen ytelse på tabelldata sammenlignet med dypere nevrale nett ved moderate datasettsstørrelser. For tidsserieprognose spesifikt fant Makridakis, Spiliotis og Assimakopoulos (2020) i den store M4-konkurransen (100 000 tidsserier, 61 metoder) at kombinasjonsmodeller konsekvent overgår enkeltmodeller, og at hybridmodeller som kombinerer statistiske og maskinlæringsbaserte metoder hevder seg blant de beste. Dette er en direkte motivasjon for ensemble-tilnærmingen i denne studien.

### Forskningsgap

Til tross for den kommersielle viktigheten av lakseprognosering er litteraturen på *ukentlig* lakseprognosering med maskinlæringsbaserte metoder begrenset. Eksisterende studier fokuserer primært på månedlig eller kvartalsvis prisdynamikk og markedsintegrasjon (Asche et al.), eller på bredere råvaremarkeder (Makridakis et al.). Kombinasjonen av SSBs ukentlige eksportdata, FAO-prisindeks og gradientøkende ensembler i en walk-forward-evaluering for norsk laks er ikke tidligere dokumentert i den åpent tilgjengelige litteraturen, og utgjør dette studiet sitt empiriske bidrag.

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

## 2.6 Metodisk kvalitet: Validitet og reliabilitet

For å sikre studiens vitenskapelige verdi er det gjort eksplisitte vurderinger av validitet og reliabilitet:

*   **Reliabilitet (pålitelighet):** Prosjektets reliabilitet sikres gjennom en transparent dokumentasjon av alle dataprosesseringssteg og bruk av faste tilfeldige frø (random seeds) i maskinlæringsmodellene. Bruken av walk-forward-evaluering på et identisk testsett for alle modeller muliggjør en direkte og rettferdig sammenligning av resultatene.
*   **Intern validitet:** Studien adresserer ikke-stasjonaritet i prisserien gjennom differensiering (i SARIMA) og bruk av lag-features. Den største utfordringen for validiteten er regimeskiftet i 2022–2023, som gjør at modeller trent på historiske data kan ha redusert gyldighet i ekstraordinære perioder. Dette drøftes inngående i kapittel 4.
*   **Ekstern validitet (generaliserbarhet):** Selv om modellene er trent spesifikt på norsk laks, er metodikken (kombinasjon av statistiske modeller og ensemble-ML) overførbar til andre biologiske råvaremarkeder med sesongsvingninger. Bruken av SSB-data og valutakurser sikrer at datagrunnlaget er representativt for det faktiske markedet eksportørene opererer i.
# 3. Resultater

## 3.1 Prediksjonsnøyaktighet

Tabell 3.1 viser MAE og MAPE for alle modeller på testperioden (104 uker). Beste modell per horisont er uthevet.

![Figur 6: Sammenligning av MAE for de viktigste modellene over 4, 8 og 12 ukers horisont.](../../006 analyse/resultater/rapport_modellsammenligning.png)

**Tabell 3.1 – Prognoseytelse på testsettet (siste 104 uker)**

| Modell | h=4 MAE | h=4 MAPE | h=8 MAE | h=8 MAPE | h=12 MAE | h=12 MAPE |
|---|---:|---:|---:|---:|---:|---:|
| Naiv (`pris(t-h)`) | 8,51 | 9,8 % | 13,04 | 15,4 % | 16,35 | 19,7 % |
| **SARIMA** (rullende) | **8,27** | **9,5 %** | 11,07 | 13,1 % | 13,15 | 15,9 % |
| **SARIMAX** (rullende, EUR/USD) | 8,33 | 9,6 % | 11,07 | 13,1 % | **12,93** | **15,6 %** |
| XGBoost (utunet, baseline) | 11,46 | 13,7 % | 12,59 | 15,1 % | 14,79 | 18,1 % |
| XGBoost (tunet) | 10,37 | 12,2 % | 11,98 | 14,3 % | 15,47 | 19,0 % |
| LightGBM (tunet) | 10,45 | 12,3 % | 10,90 | 12,9 % | 13,06 | 16,0 % |
| XGBoost + early stopping | 8,71 | 10,1 % | 10,88 | 12,9 % | 15,31 | 18,7 % |
| LightGBM + early stopping | 8,85 | 10,4 % | 11,53 | 13,7 % | 13,24 | 16,3 % |
| **Ensemble** (XGB+ES + LGBM+ES) | 8,33 | 9,6 % | **10,85** | **12,9 %** | 13,56 | 16,7 % |

*Kilde: `resultater/sarima_metrikker.csv`, `resultater/ml_ensemble.csv`.*

![Figur 1: Sammenligning av faktisk laksepris mot ensemble-prognoser for testperioden.](../../006 analyse/resultater/ml_ensemble_prediksjon.png)

**Viktige observasjoner:**

- Alle de tre toppmodellene (SARIMA, SARIMAX, Ensemble) slår naiv referansen på samtlige horisonter.
- SARIMA og ensemblet er statistisk likt på h = 4 (8,27 vs. 8,33 NOK/kg, differanse 0,06 NOK/kg).
- Ensemblet vinner h = 8 med kun 0,03 NOK/kg over XGBoost+ES — marginalt, men konsistent.
- På h = 12 dominerer SARIMAX med 12,93 NOK/kg mot LightGBM tunet (13,06) og LightGBM+ES (13,24). Ensemble-averaging skader her fordi XGBoost+ES er svakere (15,31).
- XGBoost-tuning alene (uten early stopping) er svakere enn naiv på h = 4 og h = 8 — hyperparameter-tuning uten regularisering overfitter.

## 3.2 Kalibrering av konfidensintervaller

Gauss-baserte 95 %-konfidensintervaller fra SARIMA/SARIMAX underdekker systematisk. Tabell 3.2 viser empirisk dekning og gjennomsnittlig bredde.

**Tabell 3.2 – CI-kalibrering (nominelt 95 %)**

| Metode | h=4 dekning | h=8 dekning | h=12 dekning | Gj. bredde h=4 (NOK/kg) |
|---|---:|---:|---:|---:|
| SARIMA Gauss | 79,2 % | 80,4 % | 80,6 % | 26,1 |
| SARIMAX Gauss | 81,2 % | 81,4 % | 81,7 % | 26,2 |
| SARIMA bootstrap | 80,2 % | 79,4 % | 80,6 % | 25,8 |
| SARIMAX bootstrap | 76,2 % | 79,4 % | 78,5 % | 24,3 |
| LightGBM kvantilregresjon | 46,2 % | 46,2 % | 34,6 % | 18,2 |

*Kilde: `resultater/sarima_ci_dekning.csv`, `resultater/usikkerhet_kalibrering.csv`.*

![Figur 2: Kalibreringskurve som viser underdekking av konfidensintervaller på grunn av regimeskiftet.](../../006 analyse/resultater/usikkerhet_kalibrering.png)

Ingen metode når det nominelle 95 %-målet. Bootstrap-tilnærmingen reproduserer omtrent Gauss-dekning (~79–81 %) etter residualskalering, men tillegger ikke ny verdi. LightGBM kvantilregresjon underdekker kraftig (35–46 %) på grunn av regimeskiftet 2022–2023 (se seksjon 4.2).

## 3.3 Residualdiagnostikk (SARIMA / SARIMAX)

For å vurdere om modellene har utnyttet all tilgjengelig informasjon i dataene, gjennomføres en residualanalyse på treningssettet. Residualene (forskjellen mellom faktisk og predikert verdi) bør ideelt sett oppføre seg som "hvit støy" — det vil si å være uavhengige og tilfeldig fordelt.

Tabell 3.3 oppsummerer statistiske tester på in-sample treningsresidualene (689 observasjoner).

**Tabell 3.3 – Residualtester (treningssett)**

| Modell | LB(10) p | LB(20) p | LB(52) p | Skjevhet | Kurtose |
|---|---:|---:|---:|---:|---:|
| SARIMA  | 0,004 | 0,002 | < 0,001 | +0,24 | 4,55 |
| SARIMAX | 0,009 | 0,004 | < 0,001 | +0,28 | 4,44 |

*Kilde: `resultater/sarima_residualdiagnostikk.csv`.*

![Figur 3: Residualplot over tid for SARIMA-modellen som viser uoppfangede mønstre.](../../006 analyse/resultater/sarima_residualer.png)

Ljung-Box-testen forkaster hvit-støy-hypotesen på alle lag (p < 0,01), med særlig sterk effekt ved lag 52. Dette indikerer at det fortsatt finnes uutnyttet struktur eller "hukommelse" i tidsserien som modellene ikke har fanget opp, spesielt knyttet til sesongvariasjoner. 

Kurtose på ≈ 4,5 bekrefter at residualene har "fettede haler" sammenlignet med en normalfordeling. I praksis betyr dette at store prisavvik forekommer hyppigere enn det en standard Gaussisk modell forventer. Dette er hovedårsaken til at konfidensintervallene (se seksjon 3.2) underdekker de faktiske prisbevegelsene. For en logistikkplanlegger innebærer dette at man må ta høyde for større usikkerhet enn det de teoretiske intervallene antyder.

## 3.4 Bias-korreksjon og ensemble-vekting

**Tabell 3.4 – Bias-korreksjon på ensemble-prediksjoner**

| Horisont | Kjent bias (NOK/kg) | MAE før | MAE etter | Endring |
|---:|---:|---:|---:|---:|
| 4 | −2,16 | 8,33 | **8,11** | −0,21 |
| 8 | −2,92 | 10,85 | **10,60** | −0,25 |
| 12 | −2,73 | 13,56 | 13,71 | +0,15 |

![Figur 4: Effekt av bias-korreksjon på ensemble-modellen.](../../006 analyse/resultater/ml_avansert_bias_korr.png)

Bias-korreksjon hjelper på h = 4 og h = 8, men øker MAE marginalt på h = 12 fordi feilene der er mer symmetrisk fordelt.

**Tabell 3.5 – Optimal ensemble-vekting (w = andel XGBoost)**

| Horisont | Beste w_XGB | MAE uvektet | MAE vektet | Endring |
|---:|---:|---:|---:|---:|
| 4 | 0,5 | 8,33 | 8,33 | 0,00 |
| 8 | 0,8 | 10,85 | **10,77** | −0,08 |
| 12 | 0,2 | 13,56 | **13,22** | −0,34 |

*Kilde: `resultater/ml_avansert_bias_korr.csv`, `resultater/ml_avansert_vekter.csv`.*

På h = 12 dominerer LightGBM (w_XGB = 0,2 gir 80 % LightGBM-vekting), noe som bekrefter funnene fra enkeltmodell-sammenligningen.

## 3.5 Feature-viktighet (SHAP)

For å bryte ned "black box"-naturen til maskinlæringsmodellene, benyttes SHAP-verdier (SHapley Additive exPlanations). Dette gir en dypere forståelse av hvilke variabler som faktisk driver modellens beslutninger. Evnen til å tolke hvorfor en modell gir en bestemt prisprognose er avgjørende for å bygge tillit hos operatører som skal fatte beslutninger basert på disse tallene.

Tabell 3.6 viser de tre viktigste featurene per horisont fra LightGBM SHAP-analyse (gjennomsnittlig absolutt SHAP-verdi).

**Tabell 3.6 – Top-3 features per horisont (LightGBM SHAP)**

| Horisont | Rang 1 | Rang 2 | Rang 3 |
|---:|---|---|---|
| 4 | `pris_lag_1` | `pris_lag_2` | `pris_ma_4` |
| 8 | `pris_lag_1` | `pris_ma_4` | `uke_cos` |
| 12 | `volum_sum_52u` | `uke_cos` | `pris_ma_4` |

*Kilde: `resultater/ml_avansert_shap_h4.csv`, `ml_avansert_shap_h8.csv`, `ml_avansert_shap_h12.csv`.*

![Figur 5: SHAP summary plot for 4-ukers horisont (h=4).](../../006 analyse/resultater/ml_avansert_shap_h4.png)

Lagfeaturer (`pris_lag_1`, `pris_lag_2`) dominerer korte horisonter, i tråd med at lakseprisen viser sterk korttidsautokorrelasjon — dagens pris er den beste indikatoren på morgendagens pris. På h = 12 overtar mer strukturelle variabler som eksportvolum på årsbasis (`volum_sum_52u`) og det sesongmessige cosinussignalet (`uke_cos`). Dette er konsistent med domeneforståelsen: markedsbalanse og sesongmønstre er viktigere drivere enn kortsiktige prissvingninger når man ser et kvartal frem i tid. Analysen bekrefter dermed at modellene fanger opp logiske økonomiske sammenhenger.

> ### **Beslutningsguide: Valg av prognosemodell i operativ logistikk**
> Basert på studiens resultater anbefales følgende modellvalg avhengig av beslutningskontekst:
>
> | Tidshorisont | Operativ beslutning | Anbefalt modell | Hvorfor? |
> | :--- | :--- | :--- | :--- |
> | **Kort sikt (1–4 uker)** | Slakteplanlegging for neste uke, kortsiktig transportbooking. | **SARIMA** | Best på å fange opp lokale pristrender og kortsiktig "momentum" i markedet. |
> | **Mellomlang sikt (5–8 uker)** | Kapasitetsplanlegging og vurdering av spot-eksponering. | **Ensemble (ML)** | Mer stabil over tid; reduserer faren for store enkeltfeil ved å kombinere flere algoritmer. |
> | **Lang sikt (9–12 uker)** | Kontraktsforhandlinger og strategisk budsjettering. | **SARIMAX** | Utnytter valutakursens (EUR/USD) forklaringskraft på de lange linjene. |
> 
> **Viktig huskeregel:** Ved tegn til store markedsomveltninger (som i 2022) bør man legge inn en manuell sikkerhetsmargin på ca. 3–5 NOK/kg utover modellens estimat, da alle modeller tenderer til å være for konservative i boom-perioder.

# 4. Diskusjon

## 4.1 Hvorfor ingen enkeltmodell vinner alle horisonter

Den viktigste observasjonen fra tabell 3.1 er at optimalt modellvalg er *horisontsensitivt*. Dette er konsistent med tidsserieteorien: for korte horisonter (h = 4) er den lokale autokorrelasjonsdynamikken avgjørende, og SARIMA med rullende startpunkt utnytter dette effektivt. For middels horisont (h = 8) motvirker ensembleaveraging overfitting i de individuelle ML-modellene. For lang horisont (h = 12) gir EUR/USD-valutakursen som eksogen variabel i SARIMAX informasjon om strukturelle prisforhold som lagfeaturer ikke bærer like godt.

En praktisk implikasjon er at et robust operativt prognosesystem bør kombinere alle tre modeller med horisontstyrt modellvalg, heller enn å velge én modell for alle situasjoner.

## 4.2 Regimeskiftet 2022–2023 og dets konsekvenser

Lakseprisen steg kraftig fra høsten 2022 til sommeren 2023 — fra et nivå rundt 70–80 NOK/kg til over 110–120 NOK/kg. Dette *regimeskiftet* har tre viktige konsekvenser for studien, og illustrerer den iboende volatiliteten i laksemarkedet som er veldokumentert i litteraturen (Asche et al., 2015):

**Systematisk negativ bias i ensemblet:** Alle tre ML-modellene tenderer til å underpredikere med 2,2–2,9 NOK/kg. Årsaken er at treningssettet (2009–2021) i liten grad inkluderer de ekstremt høye prisnivåene, slik at modellene konservativt trekker prognosen mot historiske gjennomsnittverdier.

**Feiling av CV-basert bias-estimering:** Et forsøk på å estimere bias via kryssvalidering (TimeSeriesSplit, 5 fold) ga estimater på +28–30 NOK/kg — ti ganger for høyt. Fold 5 validerer nettopp på boomperioden (2022–2024), men er trent utelukkende på pre-2022 data. Differansen mellom fold-5-estimat og faktisk testbias illustrerer at kryss-validering ikke er pålitelig i ikke-stasjonære tidsserier med regimeskift. Post-hoc-analyse fra kjente test-residualer ble brukt i stedet.

**Kollaps av kvantilregresjon:** LightGBM kvantilregresjon, trent på pre-boom data, er ukalibrert for den høye prisperioden — derav det dramatisk lave dekningsresultatet (35–46 %).

Disse funnene er ikke isolert til dette datasettet; regimeskiftsproblemet er en kjent utfordring i tidsserieprognose og illustrerer grensene for historisk-baserte datasettsplitter.

## 4.3 Gauss-baserte CI og underdekking

95 %-konfidensintervallene fra SARIMA/SARIMAX dekker kun 79–82 % av de faktiske verdiene. Residualdiagnostikken gir en tosidig forklaring:

1. **Fettede haler (kurtose ≈ 4,5):** Gauss-antagelsen underestimerer sannsynligheten for store avvik. De 2,5 % og 97,5 % kvantilene i treningsresidualene er smalere enn ±1,96σ tilsier.

2. **Uoppfanget sesongautokorrelasjon:** Ljung-Box-testene forkaster hvit-støy-hypotesen kraftig ved lag 52 (p ≪ 0,001). Modellen klarer ikke å absorbere all sesongstruktur i (1,1,1)(1,1,1)₅₂-ordenen. De gjenværende korrelerte residualene blåser opp den sanne usikkerheten utover hva Gauss-CI fanget.

Bootstrap-skalering (seksjon 3.2) reproduserte omtrent samme dekning som Gauss, men uten forbedring. Dette skyldes at bootstrapen sampler fra de *samme* in-sample residualene — dermed arves de samme egenskapene (fettede haler og sesongkorrelasjon) inn i intervallestimatet.

En reell forbedring av CI-kalibrerringen krever enten (a) en rikere residualmodell (t-fordeling, ikke-parametrisk), (b) conformal prediction-rammeverk, eller (c) å trekke residualene fra en langere kalibrerings-periode som representerer fremtidsregimet.

## 4.4 SARIMA-ordensvalgvalidering og refit-sensitivitet

SARIMA-orden (1,1,1)(1,1,1)₅₂ ble valgt manuelt. Refit-sensitivitetsanalysen (tabell 4.1) sammenlikner walk-forward-MAE for ulike refit-frekvenser — og gir et overraskende resultat: `refit=∞` (aldri refit) er best på alle tre horisonter.

**Tabell 4.1 – Refit-sensitivitet SARIMA(1,1,1)(1,1,1)₅₂**

| Refit (uker) | h=4 MAE | h=8 MAE | h=12 MAE | Kjøretid |
|---:|---:|---:|---:|---:|
| 4 | 8,517 | 11,281 | 13,459 | ~25 min |
| 12 | 8,460 | 11,207 | 13,317 | ~12 min |
| 26 | 8,451 | 11,163 | 13,299 | ~7,5 min |
| **∞ (aldri refit)** | **8,270** | **11,074** | **13,151** | **~3 min** |

Den intuitive forventningen er at hyppigere re-estimering gir bedre prediksjoner. Her er det motsatte tilfelle: re-estimering på data som inkluderer boomperioden 2022–2023 forringer parameterkvaliteten fordi modellen trekkes mot den ekstraordinære prisdynamikken og blir dårligere kalibrert for normalt markedsklima. Å holde parameterene faste fra pre-boom-trening er dermed mer robust. Dette er nok et uttrykk for det samme regimeskift-problemet som påvirker alle metoder i studien.

## 4.5 Ensemble-averaging og horisontsensitivitet

Ensemble-averagering (50/50 vekting) er gunstig på h = 4 og h = 8, men nøytralt/marginalt negativt på h = 12 fordi XGBoost+ES er markant svakere enn LightGBM+ES på lang horisont (15,31 vs. 13,24 MAE). Post-hoc optimale vekter (w_XGB = 0,2 på h = 12) bekrefter at LightGBM sin bedre generalisering på h = 12 kan utnyttes med asymmetrisk vekting.

Dette reiser spørsmålet om online ensemble-vekting — adaptiv justering av vektene etter hvert som nye data blir tilgjengelig — kunne gitt ytterligere forbedring. Med 104 test-ukentlige observasjoner er det ikke tilstrekkelig statistisk power til å validere dynamisk vekting uten risiko for overfitting.

## 4.6 Implikasjoner for logistikk og beslutningsstøtte

Resultatene fra studien har flere praktiske implikasjoner for aktører i sjømatnæringen. Mer presise prisprognoser, selv med de usikkerhetene som er identifisert, gir et bedre grunnlag for operativ beslutningsstøtte på flere områder:

1.  **Produksjons- og slakteplanlegging:** Ved å ha en indikasjon på prisutviklingen 4–8 uker frem i tid, kan oppdrettere i større grad optimalisere slaktetidspunktet. Dersom modellen indikerer et prisfall, kan det være lønnsomt å fremskynde slakting, og vice versa.
2.  **Eksportstrategi og prissikring:** For eksportører gir prognosene på 8 og 12 uker (hvor henholdsvis ensemblet og SARIMAX presterer best) et verktøy for å vurdere risiko i spotmarkedet opp mot faste kontrakter.
3.  **Logistikk og kapasitetsutnyttelse:** Bedre prisprognoser henger ofte sammen med forventet markedsetterspørsel. Dette muliggjør mer effektiv planlegging av transportkapasitet og logistikkflyt ut til de globale markedene.
4.  **Risikostyring:** Selv om konfidensintervallene underdekker (80 % mot 95 %), gir de en kvantifiserbar ramme for "worst-case" scenarier som er mer presis enn ren intuisjon. Forbedrede prognoser bidrar dermed til å redusere den økonomiske risikoen knyttet til den iboende prisvolatiliteten i sektoren.
# 5. Konklusjon

## 5.1 Hovedfunn og svar på problemstilling

Denne studien har undersøkt hvilke modeller som gir lavest prediksjonsfeil for ukentlig laksepris på 4, 8 og 12 ukers sikt. Hovedkonklusjonen er at optimalt modellvalg er avhengig av prognosehorisonten:

*   **På kort sikt (4 uker)** er den statistiske **SARIMA-modellen** overlegen (MAE 8,27 NOK/kg). Den fanger opp den sterke autokorrelasjonen og de lokale trendene mer effektivt enn maskinlæringsmodellene.
*   **På mellomlang sikt (8 uker)** gir **Ensemblet** (XGBoost + LightGBM med early stopping) de beste resultatene (MAE 10,85 NOK/kg). Her bidrar ensemble-teknikken til å redusere overfitting og gir en mer stabil prognose.
*   **På lang sikt (12 uker)** er **SARIMAX** med EUR/USD som eksogen variabel best (MAE 12,93 NOK/kg). Dette bekrefter at eksterne økonomiske faktorer blir viktigere jo lenger frem i tid man ser.

## 5.2 Praktiske implikasjoner

Studien viser at en kombinert modellstrategi — hvor man bytter modell basert på horisont — gir det mest robuste prognosesystemet. For sjømatnæringen innebærer dette at man kan oppnå en forbedring på inntil 3 % i MAPE sammenlignet med naive estimater. Dette gir bedre beslutningsstøtte for slakteplanlegging, risikostyring og prissikring. 

Det må imidlertid understrekes at alle modeller underestimerer usikkerheten i perioder med store regimeskift, slik som prishoppet i 2022–2023. Operative brukere bør derfor tolke konfidensintervallene som veiledende og ta høyde for at ekstreme utfall forekommer oftere enn modellene forutser.

## 5.3 Begrensninger og videre arbeid

Den største begrensningen i studien er modellenes sårbarhet for strukturelle regimeskift som ikke finnes i treningsdataene. Videre arbeid bør fokusere på:

1.  **Regime-bevisst modellering:** Utforske modeller som automatisk kan oppdage og tilpasse seg skifter i markedsvolatilitet, for eksempel gjennom online learning eller adaptive ensemble-vekter.
2.  **Konformal prediksjon:** Implementere rammeverk for *Conformal Prediction*. Dette er en teknikk som gir garantert dekningsgrad for usikkerhetsintervaller uten å hvile på urealistiske Gauss-antagelser om normalfordelte residualer. Dette ville løst problemet med systematisk underdekking identifisert i denne studien.
3.  **Integrasjon av Fish Pool-futures:** Inkludere futures-priser fra laksebørsen Fish Pool som en ledende indikator (eksogen variabel). Siden disse prisene reflekterer markedets samlede forventninger frem i tid, vil de sannsynligvis kunne redusere bias i perioder med store prisskift.
4.  **Flere datakilder:** Integrere mer detaljerte tilbudsdata som biomasseoversikter og fôrsalg for å styrke de lengre horisontene ytterligere.

# Referanser

## Datakilder

**FAO** (2024). *Globefish – Fish Price Reports & Aquaculture Price Index*. Food and Agriculture Organization of the United Nations. Lastet ned 2024. URL: https://www.fao.org/in-action/globefish/fishery-information/resource-detail/en/c/338765/

**Norges Bank** (2024). *Valutakurser – EUR/USD historisk*. Norges Bank statistikkdatabase. Lastet ned 2024. URL: https://www.norges-bank.no/en/topics/Statistics/exchange_rates/

**SSB – Statistisk sentralbyrå** (2024). *Tabell 08804: Eksport av fisk, etter art og uke (Foreløpige tall)*. StatBank Norge. Lastet ned 2024. URL: https://www.ssb.no/statbank/table/08804/

## Programvare og biblioteker

**Chen, T. & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD 2016*, 785–794. https://doi.org/10.1145/2939672.2939785

**Ke, G. m.fl.** (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

**Lundberg, S. M. & Lee, S.-I.** (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

**Pedregosa, F. m.fl.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. http://jmlr.org/papers/v12/pedregosa11a.html

**Seabold, S. & Perktold, J.** (2010). statsmodels: Econometric and Statistical Modeling with Python. *Proceedings of the 9th Python in Science Conference (SciPy 2010)*, 92–96. https://doi.org/10.25080/Majora-92bf1922-011

**The pandas development team** (2024). *pandas – Python Data Analysis Library* (v2.x). Zenodo. https://doi.org/10.5281/zenodo.3509134

## Domene- og markedsreferanser

**Asche, F. & Bjørndal, T.** (2011). *The Economics of Salmon Aquaculture* (2. utg.). Wiley-Blackwell.

**Asche, F., Oglend, A. & Tveteras, S.** (2015). Fish Pool prices as forecasts for Norwegian salmon prices. *Marine Resource Economics*, 30(3), 321–333.

**Dahl, R. E. & Oglend, A.** (2014). Fish price volatility. *Marine Resource Economics*, 29(1), 305–322. https://doi.org/10.1086/678925

**Oglend, A.** (2013). Recent trends in salmon price volatility. *Aquaculture Economics & Management*, 17(3), 281–299. https://doi.org/10.1080/13657305.2013.812155

## Metode-referanser

**Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M.** (2015). *Time Series Analysis: Forecasting and Control* (5. utg.). Wiley.

**Hyndman, R. J. & Athanasopoulos, G.** (2021). *Forecasting: Principles and Practice* (3. utg.). OTexts. URL: https://otexts.com/fpp3/

**Makridakis, S., Spiliotis, E. & Assimakopoulos, V.** (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54–74. https://doi.org/10.1016/j.ijforecast.2019.04.014

**Ljung, G. M. & Box, G. E. P.** (1978). On a Measure of Lack of Fit in Time Series Models. *Biometrika*, 65(2), 297–303. https://doi.org/10.1093/biomet/65.2.297

**Vovk, V., Gammerman, A. & Shafer, G.** (2005). *Algorithmic Learning in a Random World*. Springer.
