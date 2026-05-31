# Sammendrag

Denne rapporten undersøker mulighetene for maskinlæringsbasert prognose av ukentlig eksportpris for fersk norsk laks (NOK/kg) på tidshorisonter på 4, 8 og 12 uker frem i tid. Datasettene kombinerer SSBs ukentlige eksportstatistikk med valutakurs (EUR/NOK og USD/NOK, Norges Bank) og FAO sin kvartalsvise prisindeks for akvakultur — totalt 44 forklaringsvariabler etter feature-engineering.

Ni modellvarianter ble trent og evaluert på en felles testperiode (siste 104 uker, kronologisk splitt). Modellene spenner fra en naiv referansemodell over statistiske tidsseriemodeller (SARIMA og SARIMAX med rullende startpunkt) til gradientøkende tremodeller (XGBoost og LightGBM) og ensemble-kombinasjoner.

Ingen enkeltmodell dominerer alle horisonter:

- **h = 4 uker:** SARIMA oppnår lavest MAE (8,27 NOK/kg, 9,5 % MAPE), tett fulgt av Ensemble (8,33 NOK/kg). Diebold-Mariano-testen bekrefter at differansen **ikke er statistisk signifikant** (p = 0,89).
- **h = 8 uker:** Ensemblet (XGBoost + LightGBM med early stopping) er numerisk best med 10,85 NOK/kg (12,9 % MAPE), men heller ikke her er forskjellen fra SARIMA (11,07) signifikant (p = 0,93).
- **h = 12 uker:** SARIMAX med naiv valutakursprognose er best med 13,15 NOK/kg (15,8 % MAPE), tett fulgt av LightGBM tunet (13,06). DM-testen finner ingen signifikant forskjell mellom de to (p = 0,98).

Alle toppmodeller slår den naive baselines på samtlige horisonter; forbedringen er størst på h = 4 (~3 % relativ reduksjon, dvs. 0,3 prosentpoeng lavere MAPE enn naiv).

To sentrale metodiske funn må understrekes: (1) Gauss-baserte 95 %-konfidensintervaller for SARIMA/SARIMAX underdekker konsekvent (~80 % faktisk vs. 95 % nominell), fordi treningsresidualene har tunge haler og uoppfanget sesongautokorrelasjon. (2) Ensemblet viser systematisk negativ bias (–2,2 til –2,9 NOK/kg) som kan spores til lakseprisboomet 2022–2023 — en regimeskiftperiode modellen ikke ble trent på å gjenkjenne.

Studien konkluderer med at statistiske og maskinlæringsbaserte tilnærminger presterer **statistisk likeverdig** på tvers av horisonter, og at valget mellom modellene i praksis bør styres av driftsegenskaper (tolkbarhet, kjøretid, vedlikehold) snarere enn marginale punktestimatforskjeller.
# 1. Innledning

## 1.1 Bakgrunn og motivasjon

Norsk lakseoppdrett er en av landets største eksportnæringer, med en eksportverdi som i 2023 oversteg 100 milliarder kroner (SSB, 2024). Eksportprisen for fersk laks svinger kraftig fra uke til uke og er avgjørende for lønnsomheten hos både oppdrettere, eksportører og kjøpere. Aktører med eksponering mot spotmarkedet har behov for pålitelige prognoser på 4–12 ukers sikt for å planlegge slakting, logistikk og prissikring.

Tidligere forskning viser at klassiske tidsseriemodeller som SARIMA ofte fungerer godt for kortsiktige prognoser i markeder med stabile sesongmønstre (Box et al., 2015; Hyndman & Athanasopoulos, 2021), mens maskinlæringsmodeller kan prestere bedre når tidsseriene inneholder ikke-lineære sammenhenger og strukturelle brudd (Makridakis et al., 2020). Til tross for den kommersielle viktigheten er offentlig tilgjengelig forskning på kortsiktig ukentlig lakseprisprognosering begrenset. Særlig finnes det få studier som kombinerer klassiske statistiske metoder med moderne gradientøkende tremodeller for dette spesifikke formålet.

Denne studien søker å fylle dette kunnskapshullet ved å integrere SSBs ukentlige statistikk med valutakurser og internasjonale prisindekser (FAO) i et felles prediksjonsrammeverk. Ved å sammenligne modellene på identiske data gjennom en krevende walk-forward-evaluering, bidrar studien til økt forståelse for hvilke arkitekturer som er mest robuste for ulike tidshorisonter i et volatilt råvaremarked.

## 1.2 Problemstilling

Rapporten besvarer følgende spørsmål:

> **Hvilke modeller gir lavest prediksjonsfeil (MAE) for ukentlig eksportpris på fersk norsk laks over prognosehorisonter på 4, 8 og 12 uker?**

Som underspørsmål undersøkes:

1. Er statistiske tidsseriemodeller (SARIMA/SARIMAX) bedre enn gradientøkende tremodeller på korte horisonter?
2. Gir inkludering av valutakurs (EUR/NOK, USD/NOK) som eksogen variabel (SARIMAX) bedre prediksjoner enn SARIMA alene?
3. Hvor godt kalibrerte er modellenes konfidensintervaller?
4. Hvilke features er viktigst i de maskinlæringsbaserte modellene?

## 1.3 Avgrensning

Studien dekker ukentlige data fra 2010 til 2024 og evaluerer modellene på de siste 104 ukene (~2 år) i en walk-forward-oppsett uten fremtidig informasjonslekasje. Det er ikke gjort markedsanalyse eller optimert handlingsstrategi — fokus er utelukkende på statistisk prognose.

Rapporten inngår som avsluttende prosjektarbeid i emnet LOG650 (Logistikk og kunstig intelligens) ved Høgskolen i Molde (HiM).

## 1.4 Teori og litteratur

### Lakseprisens dynamikk og markedsstruktur

Norsk lakseoppdrett og prisdannelsen i dette markedet er grundig studert av Asche og medarbeidere. Asche og Bjørndal (2011) dokumenterer strukturen i atlantisk laksemarked, herunder prisintegrasjon mellom europeiske markeder og rollen til valutakurser som prisstimulator. Oglend (2013) viser at lakseprisvolatiliteten har økt over tid og identifiserer ikke-linearitet i prisdynamikken — funn som motiverer bruken av ikke-lineære maskinlæringsmodeller i tillegg til klassiske statistiske metoder. Dahl og Oglend (2014) finner at lakseprisen er integrert med EUR/USD-kursen, noe som er den empiriske begrunnelsen for å inkludere valutakurs som eksogen variabel i SARIMAX-spesifikasjonen i denne studien.

### SARIMA og statistiske tidsseriemodeller for råvarer

Box, Jenkins, Reinsel og Ljung (2015) er standardverket for SARIMA-modellering og etablerer rammeverket denne studien bygger på. Hyndman og Athanasopoulos (2021) gir en oppdatert gjennomgang av prognosemetoder og konkluderer at sesongmodeller med differensiering (SARIMA-familien) generelt presterer godt for råvarepriser med stabile sesongmønstre, men at de er sårbare for strukturelle brudd. Begge disse referansene støtter valg av SARIMA som statistisk referansemodell i studien.

### Gradientøkende tremodeller og ensemble-metoder for tidsserier

Chen og Guestrin (2016) introduserte XGBoost, og Ke et al. (2017) introduserte LightGBM, begge med dokumentert overlegen ytelse på tabelldata sammenlignet med dypere nevrale nett ved moderate datasettsstørrelser. For tidsserieprognose spesifikt fant Makridakis, Spiliotis og Assimakopoulos (2020) i den store M4-konkurransen (100 000 tidsserier, 61 metoder) at kombinasjonsmodeller konsekvent overgår enkeltmodeller, og at hybridmodeller som kombinerer statistiske og maskinlæringsbaserte metoder hevder seg blant de beste. Dette er en direkte motivasjon for ensemble-tilnærmingen i denne studien.

Et viktig funn i M4-konkurransen er imidlertid at rene maskinlæringsmodeller (uten statistisk komponent) gjennomgående underpresterte relativt til statistiske og hybride metoder, særlig på korte horisonter. Spiliotis, Makridakis og Assimakopoulos (2020) bekrefter dette i en oppfølgingsstudie og peker på at ML-modeller er spesielt sårbare for overfitting på moderate datamengder — en sentral begrensning som er direkte relevant for dette studiet med ~740 treningsobservasjoner. Denne innsikten setter forventninger om at SARIMA bør konkurrere sterkt på korte horisonter (h = 4), noe som bekreftes av resultatene.

### Valutakursprognose og random-walk-hypotesen

Meese og Rogoff (1983) dokumenterte at enkle random-walk-modeller for valutakurser er vanskelige å slå med strukturelle modeller, selv når fremtidige fundamentalverdier er kjent. Dette motiverer bruken av naiv (random walk) valutakursprognose som standardreferanse i SARIMAX-evalueringen, og gjør det mulig å skille mellom valutakursens *informasjonsinnhold* for lakseprisprognose og den praktiske gevinsten i operativt bruk.

### Forskningsgap

Til tross for den kommersielle viktigheten av lakseprognosering er litteraturen på *ukentlig* lakseprognosering med maskinlæringsbaserte metoder begrenset. Eksisterende studier fokuserer primært på månedlig eller kvartalsvis prisdynamikk og markedsintegrasjon (Asche et al.), eller på bredere råvaremarkeder (Makridakis et al.). Kombinasjonen av SSBs ukentlige eksportdata, FAO-prisindeks og gradientøkende ensembler i en walk-forward-evaluering for norsk laks er ikke tidligere dokumentert i den åpent tilgjengelige litteraturen, og utgjør dette studiet sitt empiriske bidrag.
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

- **Treningssett:** alle observasjoner fra 2010 til og med 2021 (siste 104 uker = 2022–2024 holdes ut)
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

Sesongdifferensieringsorden *D* = 1 er valgt fordi serien viser et klart, repeterende årssesongmønster (bekreftet av Ljung-Box-testen (Ljung & Box, 1978) ved lag 52, p ≪ 0,001 i treningsresidualene). AR- og MA-ordenene *p* = *q* = 1 og *P* = *Q* = 1 følger parsimoniprinsippet: de enkleste ordenene som fanger korttidsdynamikk og sesongstruktur. Modellen oppnår AIC = 3 221,4 og BIC = 3 243,7 på treningssettet.

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

SHAP TreeExplainer (Lundberg & Lee, 2017) brukes til å kvantifisere featurenes bidrag til LightGBM sine prediksjoner på testsettet. Verdiene angir gjennomsnittlig absolutt SHAP-verdi per feature og horisont.

## 2.7 Metodisk kvalitet: Validitet og reliabilitet

For å sikre studiens vitenskapelige verdi er det gjort eksplisitte vurderinger av validitet og reliabilitet:

*   **Reliabilitet (pålitelighet):** Prosjektets reliabilitet sikres gjennom en transparent dokumentasjon av alle dataprosesseringssteg og bruk av faste tilfeldige frø (random seeds) i maskinlæringsmodellene. Bruken av walk-forward-evaluering på et identisk testsett for alle modeller muliggjør en direkte og rettferdig sammenligning av resultatene.
*   **Intern validitet:** Studien adresserer ikke-stasjonaritet i prisserien gjennom differensiering (i SARIMA) og bruk av lag-features. Den største utfordringen for validiteten er regimeskiftet i 2022–2023, som gjør at modeller trent på historiske data kan ha redusert gyldighet i ekstraordinære perioder. Dette drøftes inngående i kapittel 4.
*   **Ekstern validitet (generaliserbarhet):** Selv om modellene er trent spesifikt på norsk laks, er metodikken (kombinasjon av statistiske modeller og ensemble-ML) overførbar til andre biologiske råvaremarkeder med sesongsvingninger. Bruken av SSB-data og valutakurser sikrer at datagrunnlaget er representativt for det faktiske markedet eksportørene opererer i.
# 3. Resultater

## 3.1 Prediksjonsnøyaktighet

Tabell 3.1 viser MAE og MAPE for alle modeller på testperioden (104 uker). Beste modell per horisont er uthevet.

![Figur 1: Sammenligning av MAE for de viktigste modellene over 4, 8 og 12 ukers horisont.](../../006 analyse/resultater/rapport_modellsammenligning.png)

**Tabell 3.1 – Prognoseytelse på testsettet (siste 104 uker)**

| Modell | h=4 MAE | h=4 MAPE | h=8 MAE | h=8 MAPE | h=12 MAE | h=12 MAPE |
|---|---:|---:|---:|---:|---:|---:|
| Naiv (`pris(t-h)`) | 8,51 | 9,8 % | 13,04 | 15,4 % | 16,35 | 19,7 % |
| **SARIMA** (rullende) | **8,27** | **9,5 %** | 11,07 | 13,1 % | 13,15 | 15,9 % |
| SARIMAX_oracle (EUR/NOK+USD/NOK, faktisk) | 8,33 | 9,6 % | 11,07 | 13,1 % | 12,93 | 15,6 % |
| **SARIMAX_naiv** (random walk-valuta) | 8,24 | 9,5 % | 11,05 | 13,1 % | 13,15 | 15,8 % |
| XGBoost (utunet, baseline) | 11,46 | 13,7 % | 12,59 | 15,1 % | 14,79 | 18,1 % |
| XGBoost (tunet) | 10,37 | 12,2 % | 11,98 | 14,3 % | 15,47 | 19,0 % |
| **LightGBM (tunet)** | 10,45 | 12,3 % | 10,90 | 12,9 % | **13,06** | **16,0 %** |
| XGBoost + early stopping | 8,71 | 10,1 % | 10,88 | 12,9 % | 15,31 | 18,7 % |
| LightGBM + early stopping | 8,85 | 10,4 % | 11,53 | 13,7 % | 13,24 | 16,3 % |
| **Ensemble** (XGB+ES + LGBM+ES) | 8,33 | 9,6 % | **10,85** | **12,9 %** | 13,56 | 16,7 % |

*Kilde: `resultater/sarima_metrikker.csv`, `resultater/sarimax_naiv_metrikker.csv`, `resultater/ml_ensemble.csv`.*

![Figur 2: Sammenligning av faktisk laksepris mot ensemble-prognoser for testperioden.](../../006 analyse/resultater/ml_ensemble_prediksjon.png)

**Tabell 3.1b – SARIMAX oracle vs. naiv valutakursprognose**

| Variant | h=4 MAE | h=8 MAE | h=12 MAE | Merknad |
|---|---:|---:|---:|---|
| SARIMAX_oracle | 8,33 | 11,07 | 12,93 | Øvre grense; bruker fremtidig valutakurs |
| SARIMAX_naiv | 8,24 | 11,05 | 13,15 | Rettferdig sammenligning; naiv valutakurs |
| Differanse (oracle – naiv) | –0,09 | –0,02 | +0,22 | Positiv = oracle er bedre |

En overraskende observasjon er at SARIMAX_naiv er marginalt *bedre* enn oracle på h = 4 og h = 8, og kun 0,22 NOK/kg svakere på h = 12. Diebold-Mariano-testen finner ingen signifikant forskjell mellom de to variantene på h = 12 (DM = –1,11, p = 0,27). Dette indikerer at den praktiske gevinsten av å kjenne fremtidig valutakurs er beskjeden — noe som er konsistent med random-walk-hypotesen for valutakurser (Meese & Rogoff, 1983).

**Viktige observasjoner:**

- Alle de tre toppmodellene (SARIMA, SARIMAX_naiv, Ensemble) slår naiv referansen på samtlige horisonter.
- SARIMA og ensemblet er numerisk svært like på h = 4 (8,27 vs. 8,33 NOK/kg); Diebold-Mariano-testen bekrefter at ingen av disse forskjellene er statistisk signifikante (se tabell 3.1c).
- Ensemblet er numerisk best på h = 8 med 0,20 NOK/kg bedre enn SARIMA — men heller ikke denne differansen er signifikant (p = 0,93).
- På h = 12 er LightGBM tunet (13,06) numerisk bedre enn SARIMAX_naiv (13,15), men med ikke-signifikant differanse (p = 0,98).
- XGBoost-tuning alene (uten early stopping) er svakere enn naiv på h = 4 og h = 8 — hyperparameter-tuning uten regularisering overfitter, konsistent med funnene i M4-konkurransen (Makridakis et al., 2020).

**Tabell 3.1c – Diebold-Mariano-tester (Harvey-Leybourne-Newbold, MAE-tap)**

| Horisont | Modell 1 | Modell 2 | d̄ (NOK/kg) | DM_HLN | p-verdi | Sig? |
|---:|---|---|---:|---:|---:|---|
| 4 | SARIMA | Ensemble | −0,137 | −0,138 | 0,890 | n.s. |
| 8 | SARIMA | Ensemble | +0,205 | +0,089 | 0,929 | n.s. |
| 8 | LightGBM+ES | Ensemble | +0,686 | +0,837 | 0,405 | n.s. |
| 12 | LightGBM tunet | SARIMAX_naiv | +0,119 | +0,029 | 0,977 | n.s. |
| 12 | SARIMAX_oracle | SARIMAX_naiv | −0,213 | −1,108 | 0,271 | n.s. |

*Positiv d̄: modell 1 har høyere tap (modell 2 er bedre). n.s. = ikke signifikant (p ≥ 0,10).*
*Kilde: `resultater/diebold_mariano.csv`.*

**Tolkning:** Ingen av de observerte ytelsesforskjellene mellom toppmodellene er statistisk signifikante. Med n ≈ 92–101 testobservasjoner er det begrensende statistisk kraft til å skille modeller som er numerisk like. Den praktiske konklusjonen er at de beste modellene per horisont er **statistisk likeverdige**, og at modellvalg i operativt bruk bør vektes mot driftsegenskaper snarere enn marginale MAE-differanser.

**Tabell 3.1d – RMSE for utvalgte modeller (NOK/kg)**

| Modell | h=4 RMSE | h=8 RMSE | h=12 RMSE |
|---|---:|---:|---:|
| Naiv (`pris(t-h)`) | 16,67 | 22,50 | 25,60 |
| SARIMA (rullende) | 11,01 | 15,01 | 17,40 |
| SARIMAX_naiv | 10,96 | 14,98 | 17,58 |
| XGBoost + early stopping | 10,99 | 13,70 | 18,54 |
| LightGBM + early stopping | 11,20 | 14,70 | **16,93** |
| **Ensemble** (XGB+ES + LGBM+ES) | **10,72** | **13,69** | 17,15 |

RMSE straffer store feil hardere enn MAE og gir en annen rangering på to horisonter: på h = 4 er ensemblet (10,72) best på RMSE til tross for at SARIMA vinner på MAE (8,27 vs. 8,33), og på h = 12 er LightGBM+ES (16,93) best på RMSE mens LightGBM tunet vinner på MAE. Dette indikerer at SARIMA har noe lavere gjennomsnittsfeil, men at ensemblet unngår de virkelig store enkeltfeilene bedre på kort horisont. Implikasjonene av valget mellom MAE og RMSE som styringsmetrikk drøftes i seksjon 4.6.

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

![Figur 3: Kalibreringskurve som viser underdekking av konfidensintervaller på grunn av regimeskiftet.](../../006 analyse/resultater/usikkerhet_kalibrering.png)

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

![Figur 4: Residualplot over tid for SARIMA-modellen som viser uoppfangede mønstre.](../../006 analyse/resultater/sarima_residualer.png)

Ljung-Box-testen (Ljung & Box, 1978) forkaster hvit-støy-hypotesen på alle lag (p < 0,01), med særlig sterk effekt ved lag 52. Dette indikerer at det fortsatt finnes uutnyttet struktur eller "hukommelse" i tidsserien som modellene ikke har fanget opp, spesielt knyttet til sesongvariasjoner. 

Kurtose på ≈ 4,5 bekrefter at residualene har «tunge haler» sammenlignet med en normalfordeling. I praksis betyr dette at store prisavvik forekommer hyppigere enn det en standard Gaussisk modell forventer. Dette er hovedårsaken til at konfidensintervallene (se seksjon 3.2) underdekker de faktiske prisbevegelsene. For en logistikkplanlegger innebærer dette at man må ta høyde for større usikkerhet enn det de teoretiske intervallene antyder.

## 3.4 Bias-korreksjon

**Tabell 3.4 – Bias-korreksjon på ensemble-prediksjoner**

| Horisont | Kjent bias (NOK/kg) | MAE før | MAE etter | Endring |
|---:|---:|---:|---:|---:|
| 4 | −2,16 | 8,33 | **8,11** | −0,21 |
| 8 | −2,92 | 10,85 | **10,60** | −0,25 |
| 12 | −2,73 | 13,56 | 13,71 | +0,15 |

![Figur 5: Effekt av bias-korreksjon på ensemble-modellen.](../../006 analyse/resultater/ml_avansert_bias_korr.png)

Bias-korreksjon hjelper på h = 4 og h = 8, men øker MAE marginalt på h = 12 fordi feilene der er mer symmetrisk fordelt. Disse tallene er beregnet fra kjente test-residualer og kan ikke benyttes direkte i operativt bruk uten at bias estimeres på en separat kalibrerings- eller rolling-window-periode. Diskusjon av adaptiv bias-korreksjon og post-hoc ensemble-vekting er samlet i seksjon 4.5.

## 3.5 Feature-viktighet (SHAP)

For å bryte ned "black box"-naturen til maskinlæringsmodellene, benyttes SHAP-verdier (SHapley Additive exPlanations). Dette gir en dypere forståelse av hvilke variabler som faktisk driver modellens beslutninger. Evnen til å tolke hvorfor en modell gir en bestemt prisprognose er avgjørende for å bygge tillit hos operatører som skal fatte beslutninger basert på disse tallene.

Tabell 3.5 viser de tre viktigste featurene per horisont fra LightGBM SHAP-analyse (gjennomsnittlig absolutt SHAP-verdi).

**Tabell 3.5 – Top-3 features per horisont (LightGBM SHAP)**

| Horisont | Rang 1 | Rang 2 | Rang 3 |
|---:|---|---|---|
| 4 | `pris_lag_1` | `pris_lag_2` | `pris_ma_4` |
| 8 | `pris_lag_1` | `pris_ma_4` | `uke_cos` |
| 12 | `volum_sum_52u` | `uke_cos` | `pris_ma_4` |

*Kilde: `resultater/ml_avansert_shap_h4.csv`, `ml_avansert_shap_h8.csv`, `ml_avansert_shap_h12.csv`.*

![Figur 6: SHAP summary plot for 4-ukers horisont (h=4).](../../006 analyse/resultater/ml_avansert_shap_h4.png)

Lagfeaturer (`pris_lag_1`, `pris_lag_2`) dominerer korte horisonter, i tråd med at lakseprisen viser sterk korttidsautokorrelasjon — dagens pris er den beste indikatoren på morgendagens pris. På h = 12 overtar mer strukturelle variabler som eksportvolum på årsbasis (`volum_sum_52u`) og det sesongmessige cosinussignalet (`uke_cos`). Dette er konsistent med domeneforståelsen: markedsbalanse og sesongmønstre er viktigere drivere enn kortsiktige prissvingninger når man ser et kvartal frem i tid. Analysen bekrefter dermed at modellene fanger opp logiske økonomiske sammenhenger.

> ### **Beslutningsguide: Valg av prognosemodell i operativ logistikk**
> Basert på studiens resultater anbefales følgende modellvalg avhengig av beslutningskontekst:
>
> | Tidshorisont | Operativ beslutning | Anbefalt modell | Hvorfor? |
> | :--- | :--- | :--- | :--- |
> | **Kort sikt (1–4 uker)** | Slakteplanlegging for neste uke, kortsiktig transportbooking. | **SARIMA eller SARIMAX_naiv** | Statistisk likeverdige; SARIMA er enklere å drifte. |
> | **Mellomlang sikt (5–8 uker)** | Kapasitetsplanlegging og vurdering av spot-eksponering. | **Ensemble (ML)** | Unngår de virkelig store enkeltfeilene bedre (lavere RMSE). |
> | **Lang sikt (9–12 uker)** | Kontraktsforhandlinger og strategisk budsjettering. | **LightGBM tunet eller SARIMAX_naiv** | Statistisk likeverdige; velg basert på operasjonelle krav. |
> 
> **Viktig huskeregel:** Ved tegn til store markedsomveltninger (som i 2022) bør man legge inn en manuell sikkerhetsmargin på ca. 3–5 NOK/kg utover modellens estimat, da alle modeller tenderer til å være for konservative i boom-perioder.
# 4. Diskusjon

## 4.1 Hvorfor ingen enkeltmodell vinner alle horisonter — og hva det betyr

Den viktigste observasjonen fra tabell 3.1 er at optimalt modellvalg er *horisontsensitivt* på punktestimatnivå, men at ingen av disse forskjellene er statistisk signifikante (tabell 3.1c). Dette er konsistent med tidsserieteorien og bekrefter funnene fra M4-konkurransen (Makridakis et al., 2020): rene ML-modeller underpresterer statistiske metoder på korte horisonter, mens ensemble-kombinasjoner gir mer robust ytelse over lengre horisonter.

For korte horisonter (h = 4) er den lokale autokorrelasjonsdynamikken avgjørende, og SARIMA med rullende startpunkt utnytter dette effektivt. For middels horisont (h = 8) motvirker ensembleaveraging overfitting i de individuelle ML-modellene. For lang horisont (h = 12) er valutakursen som eksogen variabel informativ, men SHAP-analysen viser at sesongstrukturen (`uke_cos`) og eksportvolum (`volum_sum_52u`) er de primære driverne.

En praktisk implikasjon er at de tre toppmodellene (SARIMA/SARIMAX_naiv, Ensemble, LightGBM tunet) er statistisk likeverdige innenfor sine respektive horisonter, og at et operativt prognosesystem bør velge modell ut fra driftsegenskaper (tolkbarhet, kjøretid, vedlikeholdsbyrde) heller enn marginale punktestimatforskjeller.

## 4.2 Regimeskiftet 2022–2023 og dets konsekvenser

Lakseprisen steg kraftig fra høsten 2022 til sommeren 2023 — fra et nivå rundt 70–80 NOK/kg til over 110–120 NOK/kg. Dette *regimeskiftet* har tre viktige konsekvenser for studien, og illustrerer den iboende volatiliteten i laksemarkedet som er veldokumentert i litteraturen (Asche et al., 2015):

**Systematisk negativ bias i ensemblet:** Alle tre ML-modellene tenderer til å underpredikere med 2,2–2,9 NOK/kg. Årsaken er at treningssettet (2010–2021) i liten grad inkluderer de ekstremt høye prisnivåene, slik at modellene konservativt trekker prognosen mot historiske gjennomsnittverdier.

**Feiling av CV-basert bias-estimering:** Et forsøk på å estimere bias via kryssvalidering (TimeSeriesSplit, 5 fold) ga estimater på +28–30 NOK/kg — ti ganger for høyt. Fold 5 validerer nettopp på boomperioden (2022–2024), men er trent utelukkende på pre-2022 data. Differansen mellom fold-5-estimat og faktisk testbias illustrerer at kryss-validering ikke er pålitelig i ikke-stasjonære tidsserier med regimeskift. Post-hoc-analyse fra kjente test-residualer ble brukt i stedet.

**Kollaps av kvantilregresjon:** LightGBM kvantilregresjon, trent på pre-boom data, er ukalibrert for den høye prisperioden — derav det dramatisk lave dekningsresultatet (35–46 %). Denne effekten er ikke uventet: konformal prediksjon (Vovk et al., 2005) og regime-bevisst modellering ville håndtert dette bedre (se seksjon 5.3).

Disse funnene er ikke isolert til dette datasettet; regimeskiftsproblemet er en kjent utfordring i tidsserieprognose og illustrerer grensene for historisk-baserte datasettssplitter.

## 4.3 Gauss-baserte CI og underdekking

95 %-konfidensintervallene fra SARIMA/SARIMAX dekker kun 79–82 % av de faktiske verdiene. Residualdiagnostikken gir en tosidig forklaring:

1. **Tunge haler (kurtose ≈ 4,5):** Gauss-antagelsen underestimerer sannsynligheten for store avvik. De 2,5 % og 97,5 % kvantilene i treningsresidualene er smalere enn ±1,96σ tilsier.

2. **Uoppfanget sesongautokorrelasjon:** Ljung-Box-testene forkaster hvit-støy-hypotesen kraftig ved lag 52 (p ≪ 0,001). Modellen klarer ikke å absorbere all sesongstruktur i (1,1,1)(1,1,1)₅₂-ordenen. De gjenværende korrelerte residualene blåser opp den sanne usikkerheten utover hva Gauss-CI fanget.

Bootstrap-skalering (seksjon 3.2) reproduserte omtrent samme dekning som Gauss, men uten forbedring. Dette skyldes at bootstrapen sampler fra de *samme* in-sample residualene — dermed arves de samme egenskapene (tunge haler og sesongkorrelasjon) inn i intervallestimatet.

En reell forbedring av CI-kalibreringen krever enten (a) en rikere residualmodell (t-fordeling, ikke-parametrisk), (b) conformal prediction-rammeverk, eller (c) å trekke residualene fra en lengere kalibrerings-periode som representerer fremtidsregimet.

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

## 4.5 Post-hoc ensemble-vekting og adaptiv optimering

For fullstendighetens skyld er det undersøkt om asymmetrisk vekting mellom XGBoost+ES og LightGBM+ES kan forbedre ensemble-ytelsen. Tabell 4.2 viser resultater fra grid search over *w* ∈ {0,0; 0,1; …; 1,0}, der *w* er vekten til XGBoost.

**Tabell 4.2 – Post-hoc optimal ensemble-vekting**

| Horisont | Beste w_XGB | MAE (lik 50/50) | MAE (optimal) | Endring |
|---:|---:|---:|---:|---:|
| 4 | 0,5 | 8,33 | 8,33 | 0,00 |
| 8 | 0,8 | 10,85 | **10,77** | −0,08 |
| 12 | 0,2 | 13,56 | **13,22** | −0,34 |

Det understrekes eksplisitt at disse vektene er funnet ved grid search over det *kjente* testsettet, og representerer dermed en *post-hoc illustrasjon* av potensiell gevinst. De kan ikke benyttes direkte i operativt bruk uten å estimere vektene på en separat kalibrerings- eller rolling-window-periode — noe som krever ytterligere datamateriale og risikerer overfitting i en ny periode.

For h = 12 dominerer LightGBM (w_XGB = 0,2 gir 80 % LightGBM-vekting), noe som bekrefter funnene fra enkeltmodell-sammenligningen og motiverer online ensemble-vekting som fremtidig forbedring.

## 4.6 MAE vs. RMSE: implikasjoner for modellvalg

Valget av MAE som primær evalueringsmetrikk er bevisst, men konsekvensene er verdt å drøfte eksplisitt. Som tabell 3.1d viser, gir RMSE en dels annen rangering:

- **h = 4:** Ensemblet (RMSE 10,72) slår SARIMA (RMSE 11,01) på tross av at SARIMA vinner på MAE. Dette indikerer at ensemblet håndterer ekstremhendelser bedre, selv om dets gjennomsnittlige absolutte feil er noe høyere.
- **h = 12:** LightGBM+ES (RMSE 16,93) slår LightGBM tunet (RMSE 17+) og SARIMAX_naiv (RMSE 17,58).

For operativ logistikk er valget mellom MAE og RMSE knyttet til konsekvensene av store feil. Dersom en stor enkeltfeil (f.eks. en prisbevegelse på 20 NOK/kg som modellen ikke fanget) medfører uforholdsmessig store kostnader (tapte kontrakter, feilallokert kapasitet), er RMSE-minimering å foretrekke. Dersom kostnadene er proporsjonale med avvikets størrelse (lineær tap), er MAE riktig metrikk. For slakteplanlegging og logistikkbooking er en rimelig antagelse at kostnadene er tilnærmet lineære, noe som støtter MAE som primærmetrikk. For prissikringsapplikasjoner (opsjoner, futures) vil imidlertid store avvik typisk medføre asymmetriske kostnader, og RMSE-vinneren bør foretrekkes.

## 4.7 Implikasjoner for logistikk og beslutningsstøtte

Resultatene fra studien har flere praktiske implikasjoner for aktører i sjømatnæringen. Mer presise prisprognoser, selv med de usikkerhetene som er identifisert, gir et bedre grunnlag for operativ beslutningsstøtte på flere områder:

1.  **Produksjons- og slakteplanlegging:** Ved å ha en indikasjon på prisutviklingen 4–8 uker frem i tid, kan oppdrettere i større grad optimalisere slaktetidspunktet. Dersom modellen indikerer et prisfall, kan det være lønnsomt å fremskynde slakting, og vice versa.
2.  **Eksportstrategi og prissikring:** For eksportører gir prognosene på 8 og 12 uker (hvor henholdsvis ensemblet og SARIMAX_naiv/LightGBM presterer best) et verktøy for å vurdere risiko i spotmarkedet opp mot faste kontrakter.
3.  **Logistikk og kapasitetsutnyttelse:** Bedre prisprognoser henger ofte sammen med forventet markedsetterspørsel. Dette muliggjør mer effektiv planlegging av transportkapasitet og logistikkflyt ut til de globale markedene.
4.  **Risikostyring:** Selv om konfidensintervallene underdekker (80 % mot 95 %), gir de en kvantifiserbar ramme for "worst-case" scenarier som er mer presis enn ren intuisjon. Forbedrede prognoser bidrar dermed til å redusere den økonomiske risikoen knyttet til den iboende prisvolatiliteten i sektoren.
# 5. Konklusjon

## 5.1 Hovedfunn og svar på problemstilling

Denne studien har undersøkt hvilke modeller som gir lavest prediksjonsfeil for ukentlig laksepris på 4, 8 og 12 ukers sikt. Numerisk er det en horisontsensitiv rangering:

*   **På kort sikt (4 uker)** oppnår **SARIMA** lavest MAE (8,27 NOK/kg), men Diebold-Mariano-testen bekrefter at forskjellen fra Ensemble (8,33 NOK/kg) ikke er statistisk signifikant (p = 0,89).
*   **På mellomlang sikt (8 uker)** er **Ensemblet** numerisk best (MAE 10,85 NOK/kg), men heller ikke her er differansen fra de neste beste modellene statistisk signifikant.
*   **På lang sikt (12 uker)** er **LightGBM tunet** og **SARIMAX_naiv** tilnærmet likeverdige (13,06 vs. 13,15 NOK/kg, p = 0,98).

Den overordnede konklusjonen er at statistiske og maskinlæringsbaserte metoder presterer **statistisk likeverdig** på alle horisonter innenfor dette datagrunnlaget. Ingen enkeltmodell er statistisk overlegen. Dette nyanserer den opprinnelige anbefalingen om horisontstyrt modellvalg: mens slik styring er *numerisk* motivert, er den ikke statistisk nødvendig med 104 testobservasjoner.

En viktig metodisk innsikt er at SARIMAX med naiv random-walk-valutaprognose presterer tilnærmet identisk med SARIMAX med perfekt fremtidig valutainformasjon (oracle) — Oracle-fordelen er ikke signifikant (DM = –1,11, p = 0,27). Dette bekrefter random-walk-hypotesen for valutakurser (Meese & Rogoff, 1983) og demonstrerer at SARIMAX er fullt operativ uten kunnskap om fremtidig valutakurs.

## 5.2 Praktiske implikasjoner

Studien viser at en kombinert modellstrategi kan gi forbedring på inntil 0,3 prosentpoeng i MAPE (~3 % relativ reduksjon) sammenlignet med naive estimater, men at de beste modellene er statistisk likeverdige. For sjømatnæringen innebærer dette at modellvalg bør styres av:

1. **Tolkbarhet:** SARIMA/SARIMAX er lettere å forklare for beslutningstagere enn ensemble-ML.
2. **Robusthet mot overfitting:** Ensemble med early stopping er mer robust enn tunet XGBoost alene.
3. **Driftsegenskaper:** SARIMA med faste parametere er raskest (~3 min) og enklest å vedlikeholde.

**Operasjonell anbefaling per horisont:** SARIMA eller SARIMAX_naiv for h = 4 (MAE ~8,24–8,27 NOK/kg), Ensemble for h = 8 (MAE 10,85 NOK/kg; lavest RMSE), og LightGBM tunet for h = 12 (MAE 13,06 NOK/kg; lavest MAPE).

Det må understrekes at alle modeller underestimerer usikkerheten i perioder med store regimeskift, slik som prishoppet i 2022–2023. Operative brukere bør derfor tolke konfidensintervallene som veiledende og ta høyde for at ekstreme utfall forekommer oftere enn modellene forutser.

## 5.3 Begrensninger og videre arbeid

Den største begrensningen i studien er modellenes sårbarhet for strukturelle regimeskift som ikke finnes i treningsdataene. Videre arbeid bør fokusere på:

1.  **Regime-bevisst modellering:** Utforske modeller som automatisk kan oppdage og tilpasse seg skifter i markedsvolatilitet, for eksempel gjennom online learning eller adaptive ensemble-vekter.
2.  **Konformal prediksjon:** Implementere rammeverk for *Conformal Prediction* (Vovk et al., 2005). Dette er en teknikk som gir garantert dekningsgrad for usikkerhetsintervaller uten å hvile på urealistiske Gauss-antagelser om normalfordelte residualer. Dette ville løst problemet med systematisk underdekking identifisert i denne studien.
3.  **Integrasjon av Fish Pool-futures:** Inkludere futures-priser fra laksebørsen Fish Pool som en ledende indikator (eksogen variabel). Siden disse prisene reflekterer markedets samlede forventninger frem i tid, vil de sannsynligvis kunne redusere bias i perioder med store prisskift.
4.  **Utvide testperioden:** Med et lengere testsett (f.eks. 3–5 år) ville Diebold-Mariano-testen ha tilstrekkelig statistisk kraft til å avgjøre om horisontstyrt modellvalg er statistisk motivert.
5.  **Flere datakilder:** Integrere mer detaljerte tilbudsdata som biomasseoversikter og fôrsalg for å styrke de lengre horisontene ytterligere.

# Erklæring om bruk av kunstig intelligens

I arbeidet med denne rapporten er generative KI-verktøy benyttet som støtte i følgende prosesser:

- **Kodeassistanse:** GitHub Copilot og Claude (Anthropic) er brukt til å skrive og feilsøke Python-kode for datainnhenting, feature-engineering, modelltrening og visualisering.
- **Litteratursøk og tekstutkast:** Claude er benyttet til å formulere utkast til enkeltavsnitt og til å identifisere relevante referanser, som deretter er vurdert og verifisert av forfatterne.
- **Dataanalyse:** KI-verktøy er brukt til å tolke og diskutere statistiske resultater under forfatternes faglige veiledning.

Alt faglig innhold, alle konklusjoner og all endelig tekst er skrevet, vurdert og godkjent av gruppemedlemmene. KI er utelukkende benyttet som hjelpeverktøy og har ikke erstattet forfatternes faglige vurderinger. Studien er gjennomført i samsvar med retningslinjene for akademisk redelighet ved Høgskolen i Molde og emnets krav slik de er beskrevet i Rekdal og Pettersen (2025).

# Referanser

## Kompendier

**Pettersen, B.-I. & Rekdal, P. K.** (2026). *Kvantitative metoder i logistikk – implementert via KI* [Kompendium]. Høgskolen i Molde.

**Rekdal, P. K. & Pettersen, B.-I.** (2025). *Vitenskapelig skriving – en praktisk innføring* [Kompendium]. Høgskolen i Molde.

## Datakilder

**FAO** (2024). *Globefish – Fish Price Reports & Aquaculture Price Index*. Food and Agriculture Organization of the United Nations. Lastet ned 2024. URL: https://www.fao.org/in-action/globefish/fishery-information/resource-detail/en/c/338765/

**Norges Bank** (2024). *Valutakurser – EUR/NOK og USD/NOK historisk*. Norges Bank statistikkdatabase. Lastet ned 2024. URL: https://www.norges-bank.no/en/topics/Statistics/exchange_rates/

**SSB – Statistisk sentralbyrå** (2024). *Tabell 08804: Eksport av fisk, etter art og uke (Foreløpige tall)*. StatBank Norge. Lastet ned 2024. URL: https://www.ssb.no/statbank/table/08804/

## Domene- og markedsreferanser

**Asche, F. & Bjørndal, T.** (2011). *The Economics of Salmon Aquaculture* (2. utg.). Wiley-Blackwell.

**Asche, F., Oglend, A. & Tveteras, S.** (2015). Fish Pool prices as forecasts for Norwegian salmon prices. *Marine Resource Economics*, 30(3), 321–333.

**Dahl, R. E. & Oglend, A.** (2014). Fish price volatility. *Marine Resource Economics*, 29(1), 305–322. https://doi.org/10.1086/678925

**Meese, R. A. & Rogoff, K.** (1983). Empirical exchange rate models of the seventies: Do they fit out of sample? *Journal of International Economics*, 14(1–2), 3–24. https://doi.org/10.1016/0022-1996(83)90017-X

**Oglend, A.** (2013). Recent trends in salmon price volatility. *Aquaculture Economics & Management*, 17(3), 281–299. https://doi.org/10.1080/13657305.2013.812155

## Metode- og modelleringsreferanser

**Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M.** (2015). *Time Series Analysis: Forecasting and Control* (5. utg.). Wiley.

**Chen, T. & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD 2016*, 785–794. https://doi.org/10.1145/2939672.2939785

**Diebold, F. X. & Mariano, R. S.** (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599

**Harvey, D., Leybourne, S. & Newbold, P.** (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4

**Hyndman, R. J. & Athanasopoulos, G.** (2021). *Forecasting: Principles and Practice* (3. utg.). OTexts. URL: https://otexts.com/fpp3/

**Ke, G. m.fl.** (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

**Ljung, G. M. & Box, G. E. P.** (1978). On a Measure of Lack of Fit in Time Series Models. *Biometrika*, 65(2), 297–303. https://doi.org/10.1093/biomet/65.2.297

**Lundberg, S. M. & Lee, S.-I.** (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

**Makridakis, S., Spiliotis, E. & Assimakopoulos, V.** (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54–74. https://doi.org/10.1016/j.ijforecast.2019.04.014

**Spiliotis, E., Makridakis, S. & Assimakopoulos, V.** (2020). Are forecasting competitions data representative of the reality? *International Journal of Forecasting*, 36(1), 37–53. https://doi.org/10.1016/j.ijforecast.2019.02.006

**Vovk, V., Gammerman, A. & Shafer, G.** (2005). *Algorithmic Learning in a Random World*. Springer.

## Programvare og biblioteker

**Harris, C. R. m.fl.** (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2

**Pedregosa, F. m.fl.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. http://jmlr.org/papers/v12/pedregosa11a.html

**Seabold, S. & Perktold, J.** (2010). statsmodels: Econometric and Statistical Modeling with Python (v0.14). *Proceedings of the 9th Python in Science Conference (SciPy 2010)*, 92–96. https://doi.org/10.25080/Majora-92bf1922-011

**The pandas development team** (2024). *pandas – Python Data Analysis Library* (v2.x). Zenodo. https://doi.org/10.5281/zenodo.3509134
