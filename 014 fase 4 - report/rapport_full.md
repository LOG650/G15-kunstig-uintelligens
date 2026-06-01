# Sammendrag

Denne rapporten undersøker mulighetene for maskinlæringsbasert prognose av ukentlig eksportpris for fersk norsk laks (NOK/kg) på tidshorisonter på 4, 8 og 12 uker frem i tid. Datasettene kombinerer SSBs ukentlige eksportstatistikk med valutakurser (EUR/NOK og USD/NOK, Norges Bank) og FAO sin kvartalsvise prisindeks for akvakultur — totalt 44 forklaringsvariabler etter feature-engineering.

Ni modellvarianter ble trent og evaluert på en felles testperiode (siste 104 uker, kronologisk splitt). Modellene spenner fra en naiv referansemodell over statistiske tidsseriemodeller (SARIMA og SARIMAX med rullende startpunkt) til gradientøkende tremodeller (XGBoost og LightGBM) og ensemble-kombinasjoner.

Ingen enkeltmodell dominerer alle horisonter:

- **h = 4 uker:** SARIMA oppnår lavest MAE (8,27 NOK/kg, 9,5 % MAPE), tett fulgt av SARIMAX_naiv og ensemblet.
- **h = 8 uker:** Ensemblet (XGBoost + LightGBM med early stopping) er numerisk best med 10,85 NOK/kg (12,9 % MAPE), men er statistisk likeverdig med de andre toppmodellene.
- **h = 12 uker:** LightGBM tunet (13,06 NOK/kg) og SARIMAX_naiv (13,15 NOK/kg) er de sterkeste modellene.

Alle toppmodeller slår den naive baselines på samtlige horisonter. To sentrale metodiske funn må understrekes: (1) Gauss-baserte 95 %-konfidensintervaller for SARIMA/SARIMAX underdekker konsekvent (~80 % faktisk vs. 95 % nominell) på grunn av fettede haler i residualene. (2) Ensemblet viser systematisk negativ bias (–2,2 til –2,9 NOK/kg) som kan spores til det ekstraordinære lakseprisboomet 2022–2023 — en regimeskiftperiode modellen ikke ble trent på å gjenkjenne.

Studien konkluderer med at statistiske og maskinlæringsbaserte tilnærminger utfyller hverandre. Mens de numeriske forskjellene antyder en horisontstyrt strategi — SARIMA for korte horisonter, ensemble for mellomhorisonter og LightGBM/SARIMAX for lengre — viser statistisk testing (Diebold-Mariano) at modellene er likeverdige i ytelse. Valg av modell bør derfor i stor grad styres av operative krav til tolkbarhet og vedlikehold.

# 1. Innledning

## 1.1 Bakgrunn og motivasjon

Norsk lakseoppdrett er en av landets største eksportnæringer, med en eksportverdi som i 2023 oversteg 100 milliarder kroner (SSB, 2024). Eksportprisen for fersk laks svinger kraftig fra uke til uke og er avgjørende for lønnsomheten hos både oppdrettere, eksportører og kjøpere. Aktører med eksponering mot spotmarkedet har behov for pålitelige prognoser på 4–12 ukers sikt for å planlegge slakting, logistikk og prissikring.

Tidligere forskning viser at klassiske tidsseriemodeller som SARIMA ofte fungerer godt for kortsiktige prognoser i markeder med stabile sesongmønstre, mens maskinlæringsmodeller kan prestere bedre når tidsseriene inneholder ikke-lineære sammenhenger og strukturelle brudd. Til tross for den kommersielle viktigheten er offentlig tilgjengelig forskning på kortsiktig ukentlig lakseprisprognosering begrenset. Særlig finnes det få studier som kombinerer klassiske statistiske metoder med moderne gradientøkende tremodeller for dette spesifikke formålet.

Denne studien søker å fylle dette kunnskapshullet ved å integrere SSBs ukentlige statistikk med valutakurser og internasjonale prisindekser (FAO) i et felles prediksjonsrammeverk. Ved å sammenligne modellene på identiske data gjennom en krevende walk-forward-evaluering, bidrar studien til økt forståelse for hvilke arkitekturer som er mest robuste for ulike tidshorisonter i et volatilt råvaremarked.

## 1.2 Tidligere forskning

Norsk lakseoppdrett og prisdannelsen i dette markedet er grundig studert av Asche og medarbeidere. Asche og Bjørndal (2011) dokumenterer strukturen i atlantisk laksemarked, herunder prisintegrasjon mellom europeiske markeder og rollen til valutakurser som prisstimulator. Oglend (2013) viser at lakseprisvolatiliteten har økt over tid og identifiserer ikke-linearitet i prisdynamikken — funn som motiverer bruken av ikke-lineære maskinlæringsmodeller i tillegg til klassiske statistiske metoder. Dahl og Oglend (2014) finner at lakseprisen er integrert med EUR/NOK- og USD/NOK-kurser (ofte analysert via EUR/USD-ratioen), noe som er den empiriske begrunnelsen for å inkludere valutakurser som forklaringsvariabler i denne studien.

## 1.3 Problemstilling

Rapporten besvarer følgende spørsmål:

> **Hvilke modeller gir lavest prediksjonsfeil (MAE) for ukentlig eksportpris på fersk norsk laks over prognosehorisonter på 4, 8 og 12 uker?**

Som underspørsmål undersøkes:

1. Er statistiske tidsseriemodeller (SARIMA/SARIMAX) bedre enn gradientøkende tremodeller på korte horisonter?
2. Gir inkludering av valutakurs (EUR/NOK, USD/NOK) som forklaringsvariabel (SARIMAX) bedre prediksjoner enn SARIMA alene?
3. Hvor godt kalibrerte er modellenes konfidensintervaller?
4. Hvilke forklaringsvariabler (features) er viktigst i de maskinlæringsbaserte modellene?

## 1.4 Avgrensning

Studien dekker ukentlige data fra 2010 til 2024 og evaluerer modellene på de siste 104 ukene (~2 år, 2022–2024) i en walk-forward-oppsett uten fremtidig informasjonslekasje. Det er ikke gjort markedsanalyse eller optimert handlingsstrategi — fokus er utelukkende på statistisk prognose.

Rapporten inngår som avsluttende prosjektarbeid i emnet LOG650 (Logistikk og kunstig intelligens) ved Høgskolen i Molde (HiM).

# 2. Metode

## 2.1 Data

### Prisvariabel

Ukentlig gjennomsnittlig eksportpris for fersk hel norsk laks (NOK/kg) er hentet fra Statistisk sentralbyrå (SSB, tabell 08804). Data dekker perioden januar 2010 til mars 2024 — totalt ~741 ukentlige observasjoner i treningssettet.

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

**Stasjonaritet:** ADF- og KPSS-tester (se seksjon 2.3.2) bekrefter at serien er I(1): ikke-stasjonær på nivå, men stasjonær etter én differensiering. Dette motiverer bruken av d = 1 i SARIMA-modellen og bruken av lagverdier (i stedet for prisnivå) som forklaringsvariabler i ML-modellene.

### Forklaringsvariabler (Features)

Følgende grupper av forklaringsvariabler ble konstruert (44 kolonner totalt):

| Kilde | Variabel | Frekvens | Behandling |
|---|---|---|---|
| Norges Bank | EUR/NOK og USD/NOK spot | Daglig | Ukentlig gjennomsnitt (`eur_nok_snitt`, `usd_nok_snitt`) |
| FAO | Fiskeriprisindeks (akvakultur) | Kvartalsvis | Forward-fill til ukentlig; `fao_imputert`-flagg markerer verdier |

- **Lagverdier:** `pris_lag_1` til `pris_lag_52`
- **Glidende gjennomsnitt:** `pris_ma_4`, `pris_ma_8`, `pris_ma_12`, `pris_ma_26`, `pris_ma_52`
- **Volatilitet:** `pris_std_4`, `pris_std_8`, `pris_std_12`
- **Sesong:** `uke_sin`, `uke_cos` (sirkulær koding av ukenummer 1–52)
- **Eksportvolum:** `volum_sum_4u`, `volum_sum_12u`, `volum_sum_52u` (SSB)
- **FAO-index:** `fao_index_raw`, `fao_imputert`
- **Valuta:** `eur_nok_snitt`, `usd_nok_snitt`, `eur_usd_ratio`

De første ~52 radene inneholder NaN-verdier i lag-featurene og er fjernet fra ML-treningssettet via `dropna()`. SARIMA opererer på selve pris-tidsserien og er ikke berørt av denne trimmingen; treningssettene er dermed asymmetriske i startdato, men identiske i sluttdato og testperiode.

## 2.2 Datasplit og evalueringsprotokoll

Datamaterialet deles kronologisk:

- **Treningssett:** alle observasjoner fra 2010 til og med 2021.
- **Testsett:** de siste 104 ukene (~2 år, 2022–2024).

### Walk-forward-evaluering

For å unngå informasjonslekasje evalueres alle modeller med *walk-forward*-prognose: modellen trenes på alt frem til tidspunkt *t*, lager prognose for *t+4*, *t+8* og *t+12*, deretter forlenges treningsvinduet med én uke og prosessen gjentas. For SARIMA brukes `statsmodels append(refit=False)` — modellparametrene holder seg faste fra initial trening, men modellen oppdateres med hvert nytt datapunkt.

### Evalueringsmetrikker

- **MAE** (Mean Absolute Error, NOK/kg): tolkes direkte som gjennomsnittlig absolutt prisprediksjonsfeil. Primær metrikk.
- **MAPE** (Mean Absolute Percentage Error): muliggjør sammenligning på tvers av prisnivåer.
- **RMSE** (Root Mean Squared Error, NOK/kg): straffer store enkeltfeil hardere enn MAE; brukes som sekundær kontroll for robusthet.

MAE er valgt som primær rangeringsmetrikk fordi det er robust mot ekstreme enkeltobservasjoner, lineært tolkbart («gjennomsnittlig avvik er X NOK/kg»), og konsistent med standard prognoseevaluering i råvaremarkeder. Implikasjonene av valget mellom MAE og RMSE drøftes i seksjon 4.6.

## 2.3 Modeller

### 2.3.1 Naiv referansemodell

Prognosegrunnlag: `pris(t + h) = pris(t)`. Tilsvarer ingen modelltilpasning og er minimumsstandarden alle modeller skal slå.

### 2.3.2 SARIMA og SARIMAX

Sesongbasert autoregressiv integrert glidende gjennomsnitt (SARIMA) med orden (1, 1, 1)(1, 1, 1)₅₂, tilpasset med maksimum likelihood-estimering via `statsmodels.tsa.statespace.SARIMAX`. Sesongperiode *m* = 52 reflekterer ukentlig data.

**Begrunnelse for ordensvalget:**
Integrasjonsorden *d* = 1 er begrunnet med stasjonaritetstester. ADF-testen forkaster ikke enhetsrot på nivå (p = 0,668), men forkaster den klart etter første differensiering (p < 0,001). KPSS-testen bekrefter dette. Sesongdifferensiering *D* = 1 er valgt pga. årssesongmønsteret bekreftet av Ljung-Box-testen ved lag 52 (p ≪ 0,001).

SARIMAX inkludere valutakurser som eksogene variabler. For å gi et rettferdig sammenligningsgrunnlag evalueres SARIMAX i to varianter:
- **SARIMAX_oracle:** Bruker faktiske realiserte valutakurser for prognoseperioden (optimistisk øvre grense).
- **SARIMAX_naiv:** Bruker siste kjente valutakurs som prognose for alle fremtidige steg (random walk). Dette representerer en rettferdig operativ sammenligning (Meese & Rogoff, 1983).

### 2.3.3 XGBoost og LightGBM med early stopping

Gradient-boosted beslutningstrær tilpasset med early stopping: 20 % av treningssettet settes av som valideringssett. Hyperparametere ble søkt med `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=5)`, 60 iterasjoner, MAE-scoring. Endelige hyperparametere er dokumentert i tabell 2.2.

**Tabell 2.2 – Endelige hyperparametere (RandomizedSearchCV, MAE-scoring)**

| Parameter | XGBoost h=4 | XGBoost h=8 | XGBoost h=12 | LightGBM h=4 | LightGBM h=8 | LightGBM h=12 |
|---|---:|---:|---:|---:|---:|---:|
| `learning_rate` | 0,20 | 0,20 | 0,05 | 0,20 | 0,10 | 0,20 |
| `max_depth` | 3 | 3 | 3 | – | 8 | 10 |
| `n_estimators` (søk) | 800 | 1 000 | 600 | 400 | 1 000 | 1 000 |
| `subsample` | 0,70 | 0,90 | 1,00 | 0,80 | 1,00 | 0,60 |
| `colsample_bytree` | 0,70 | 0,60 | 1,00 | 0,80 | 0,50 | 0,80 |
| `num_leaves` (LGBM) | – | – | – | 63 | 63 | 127 |

Early stopping bestemmer det endelige antall estimatorer ved å monitere valideringstapet.

### 2.3.4 Ensemble

Ensemble-prognosen er et vektet snitt av XGBoost+ES og LightGBM+ES med lik vekting (*w* = 0,5) i den primære sammenligningen.

## 2.4 Usikkerhetskvantifisering

I tillegg til Gauss-CI fra SARIMA ble to empiriske metoder undersøkt:
- **Bootstrap:** Simulering av 2 000 fremtidsforløp basert på in-sample-residualer.
- **Kvantilregresjon (LightGBM):** Direkte estimering av 2,5 % og 97,5 % kvantiler.

## 2.5 Statistisk signifikanstesting (Diebold-Mariano)

For å avgjøre om ytelsesforskjeller mellom modeller er statistisk meningfulle, benyttes **Diebold-Mariano-testen** (Diebold & Mariano, 1995). Testen evaluerer nullhypotesen om at to modeller har lik prediksjonsnøyaktighet. En p-verdi over 0,05 betyr at vi ikke har nok bevis til å si at den ene modellen er bedre enn den andre, selv om MAE-tallene er forskjellige. Newey-West HAC-varians korrigerer for autokorrelasjon i feilene.

## 2.6 Tolkning og forklarbarhet (SHAP)

**SHAP TreeExplainer** (Lundberg & Lee, 2017) brukes til å kvantifisere bidraget fra hver forklaringsvariabel til modellens prediksjoner.

## 2.7 Metodisk kvalitet

Prosjektets **reliabilitet** sikres gjennom transparent dokumentasjon og faste tilfeldige frø. **Intern validitet** adresseres gjennom walk-forward-evaluering og stasjonaritetstester, mens utfordringer knyttet til regimeskift drøftes som begrensninger.

# 3. Resultater

## 3.1 Prediksjonsnøyaktighet

Tabell 3.1 viser MAE og MAPE for alle modeller på testperioden (104 uker). Beste modell per horisont er uthevet.

![Figur 1: Sammenligning av MAE for de viktigste modellene over 4, 8 og 12 ukers horisont.](../../006 analyse/resultater/rapport_modellsammenligning.png)

**Tabell 3.1 – Prognoseytelse på testsettet (siste 104 uker)**

| Modell | h=4 MAE | h=4 MAPE | h=8 MAE | h=8 MAPE | h=12 MAE | h=12 MAPE |
|---|---:|---:|---:|---:|---:|---:|
| Naiv (`pris(t-h)`) | 8,51 | 9,8 % | 13,04 | 15,4 % | 16,35 | 19,7 % |
| **SARIMA** (rullende) | **8,27** | **9,5 %** | 11,07 | 13,1 % | 13,15 | 15,9 % |
| SARIMAX_oracle (faktisk valuta) | 8,33 | 9,6 % | 11,07 | 13,1 % | 12,93 | 15,6 % |
| **SARIMAX_naiv** (random walk valuta) | 8,24 | 9,5 % | 11,05 | 13,1 % | 13,15 | 15,8 % |
| XGBoost (utunet, baseline) | 11,46 | 13,7 % | 12,59 | 15,1 % | 14,79 | 18,1 % |
| XGBoost (tunet) | 10,37 | 12,2 % | 11,98 | 14,3 % | 15,47 | 19,0 % |
| **LightGBM (tunet)** | 10,45 | 12,3 % | 10,90 | 12,9 % | **13,06** | **16,0 %** |
| XGBoost + early stopping | 8,71 | 10,1 % | 10,88 | 12,9 % | 15,31 | 18,7 % |
| LightGBM + early stopping | 8,85 | 10,4 % | 11,53 | 13,7 % | 13,24 | 16,3 % |
| **Ensemble** (XGB+ES + LGBM+ES) | 8,33 | 9,6 % | **10,85** | **12,9 %** | 13,56 | 16,7 % |

*Kilde: `resultater/sarima_metrikker.csv`, `resultater/ml_ensemble.csv`.*

![Figur 2: Sammenligning av faktisk laksepris mot ensemble-prognoser for testperioden.](../../006 analyse/resultater/ml_ensemble_prediksjon.png)

**Tabell 3.1b – Diebold-Mariano-tester (Harvey-Leybourne-Newbold, MAE-tap)**

| Horisont | Modell 1 | Modell 2 | d̄ (NOK/kg) | DM_HLN | p-verdi | Sig? |
|---:|---|---|---:|---:|---:|---|
| 4 | SARIMA | Ensemble | −0,137 | −0,138 | 0,890 | n.s. |
| 8 | SARIMA | Ensemble | +0,205 | +0,089 | 0,929 | n.s. |
| 12 | LightGBM tunet | SARIMAX_naiv | +0,119 | +0,029 | 0,977 | n.s. |

*Positiv d̄: modell 1 har høyere tap (modell 2 er bedre). n.s. = ikke signifikant (p ≥ 0,10).*

**Tolkning:** Ingen av de observerte ytelsesforskjellene mellom toppmodellene er statistisk signifikante. Dette betyr at selv om tallene varierer noe, er modellene i praksis **likeverdige** i ytelse på dette testsettet.

**Tabell 3.1c – RMSE for utvalgte modeller (NOK/kg)**

| Modell | h=4 RMSE | h=8 RMSE | h=12 RMSE |
|---|---:|---:|---:|
| Naiv (`pris(t-h)`) | 16,67 | 22,50 | 25,60 |
| SARIMA (rullende) | 11,01 | 15,01 | 17,40 |
| **Ensemble** (XGB+ES + LGBM+ES) | **10,72** | **13,69** | 17,15 |

RMSE-resultatene viser at **ensemblet er mer robust** mot store enkeltfeil enn SARIMA på h = 4, til tross for at SARIMA har lavere gjennomsnittlig feil (MAE).

## 3.2 Kalibrering av konfidensintervaller

Gauss-baserte 95 %-konfidensintervaller underdekker systematisk (~80 % faktisk vs. 95 % nominell).

**Tabell 3.2 – CI-kalibrering (nominelt 95 %)**

| Metode | h=4 dekning | h=8 dekning | h=12 dekning |
|---|---:|---:|---:|
| SARIMA Gauss | 79,2 % | 80,4 % | 80,6 % |
| LightGBM kvantilregresjon | 46,2 % | 46,2 % | 34,6 % |

![Figur 3: Kalibreringskurve som viser underdekking av konfidensintervaller.](../../006 analyse/resultater/usikkerhet_kalibrering.png)

## 3.3 Residualdiagnostikk (SARIMA)

Ljung-Box-testen forkaster hvit-støy-hypotesen (p < 0,01), noe som indikerer uoppfanget sesongstruktur ved lag 52. Kurtose på ≈ 4,5 bekrefter tunge haler i residualene, som forklarer CI-underdekkingen.

![Figur 4: Residualplot over tid for SARIMA-modellen.](../../006 analyse/resultater/sarima_residualer.png)

## 3.4 Bias-korreksjon

Ensemblet har en systematisk negativ bias på –2,2 til –2,9 NOK/kg. Korreksjon forbedrer MAE på h = 4 og h = 8.

![Figur 5: Effekt av bias-korreksjon på ensemble-modellen.](../../006 analyse/resultater/ml_avansert_bias_korr.png)

## 3.5 Forklaringsvariablenes viktighet (SHAP)

![Figur 6: SHAP summary plot for 4-ukers horisont (h=4).](../../006 analyse/resultater/ml_avansert_shap_h4.png)

Lagverdier (`pris_lag_1`) dominerer på kort sikt, mens strukturelle faktorer som eksportvolum (`volum_sum_52u`) og sesong (`uke_cos`) blir viktigere på h = 12.

> ### **Beslutningsguide: Valg av prognosemodell i operativ logistikk**
>
> | Tidshorisont | Operativ beslutning | Anbefalt modell | Hvorfor? |
> | :--- | :--- | :--- | :--- |
> | **Kort sikt (1–4 uker)** | Slakteplanlegging, transportbooking. | **SARIMA** | Enkel å drifte, lavest MAE. |
> | **Mellomlang sikt (5–8 uker)** | Kapasitetsplanlegging. | **Ensemble (ML)** | Robust mot store feil (lavest RMSE). |
> | **Lang sikt (9–12 uker)** | Kontraktsforhandlinger, budsjett. | **LightGBM tunet** | Numerisk best på h=12. |

# 4. Diskusjon

## 4.1 Horisontsensitivitet og statistisk likeverdighet

Selv om vi ser numeriske forskjeller (f.eks. SARIMA best på h=4, Ensemble på h=8), viser Diebold-Mariano-testene at modellene er statistisk likeverdige. Dette betyr at man i operativ drift kan velge modell basert på tolkbarhet eller kjøretid uten å ofre signifikant nøyaktighet.

## 4.2 Regimeskiftet 2022–2023

Prisboomet i 2022–2023 representerer et regimeskifte som ingen av modellene var trent på. Dette forklarer den systematiske underprediksjonen (bias) og kollapsen i kvantilregresjon. Dette illustrerer grensene for historisk-basert maskinlæring i volatile råvaremarkeder.

## 4.3 CI-underdekking

Underdekkingen av konfidensintervaller skyldes fettede haler i prisdistribusjonen og restautokorrelasjon. For logistikkplanleggere betyr dette at man må ta høyde for "svart svane"-hendelser oftere enn modellen tilsier.

## 4.4 Refit-sensitivitet

Overraskende nok ga `refit=∞` (ingen re-trening) de beste resultatene for SARIMA. Dette skyldes trolig at modellparameterne estimert på det stabile pre-2022 datasettet er mer robuste enn parametere som "forstyrres" av den ekstreme prisdynamikken i 2022–2023.

## 4.5 MAE vs. RMSE: Robusthet i logistikk

Valget mellom MAE og RMSE er kritisk. Mens SARIMA vinner på gjennomsnittsfeil (MAE) på h=4, vinner ensemblet på RMSE. For en logistikkaktør betyr dette at **ensemblet er mer pålitelig for å unngå katastrofalt store feil**, selv om det i snitt bommer litt mer enn SARIMA.

# 5. Konklusjon

Studiens hovedfunn er at både statistiske tidsseriemodeller og maskinlæringsensembler slår naive estimater, men at de beste modellene i stor grad er statistisk likeverdige. SARIMA anbefales for kortsiktig bruk pga. enkelhet, mens maskinlæringsensembler gir økt robusthet mot store avvik på mellomlang sikt. Videre arbeid bør fokusere på **Conformal Prediction** for bedre usikkerhetsestimering og inkludering av futures-priser (Fish Pool) som forklaringsvariabler.

# Erklæring om bruk av kunstig intelligens

KI-verktøy (Claude, GitHub Copilot) er brukt til kodeassistanse, tekstproduksjon og statistisk tolkning. Alt innhold er verifisert og godkjent av forfatterne, som tar fullt akademisk ansvar.

# Referanser

**Asche, F. & Bjørndal, T.** (2011). *The Economics of Salmon Aquaculture* (2nd ed.). Wiley-Blackwell.

**Dahl, R. E. & Oglend, A.** (2014). Fish price volatility. *Marine Resource Economics*, 29(1), 305–322. https://doi.org/10.1086/678925

**Diebold, F. X. & Mariano, R. S.** (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599

**Meese, R. A. & Rogoff, K.** (1983). Empirical exchange rate models of the seventies: Do they fit out of sample? *Journal of International Economics*, 14(1–2), 3–24. https://doi.org/10.1016/0022-1996(83)90017-X

**Norges Bank** (2024). *Valutakurser – EUR/NOK og USD/NOK historisk*. Norges Bank statistikkdatabase. https://www.norges-bank.no/en/topics/Statistics/exchange_rates/

**Oglend, A.** (2013). Recent trends in salmon price volatility. *Aquaculture Economics & Management*, 17(3), 281–299. https://doi.org/10.1080/13657305.2013.812155

**SSB – Statistisk sentralbyrå** (2024). *Tabell 08804: Eksport av fisk, etter art og uke*. StatBank Norge. https://www.ssb.no/statbank/table/08804/
