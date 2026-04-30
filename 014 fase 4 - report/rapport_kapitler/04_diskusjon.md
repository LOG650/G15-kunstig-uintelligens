# 4. Diskusjon

## 4.1 Hvorfor ingen enkeltmodell vinner alle horisonter

Den viktigste observasjonen fra tabell 3.1 er at optimalt modellvalg er *horisontsensitivt*. Dette er konsistent med tidsserieteorien: for korte horisonter (h = 4) er den lokale autokorrelasjonsdynamikken avgjørende, og SARIMA med rullende startpunkt utnytter dette effektivt. For middels horisont (h = 8) motvirker ensembleaveraging overfitting i de individuelle ML-modellene. For lang horisont (h = 12) gir EUR/USD-valutakursen som eksogen variabel i SARIMAX informasjon om strukturelle prisforhold som lagfeaturer ikke bærer like godt.

En praktisk implikasjon er at et robust operativt prognosesystem bør kombinere alle tre modeller med horisontstyrt modellvalg, heller enn å velge én modell for alle situasjoner.

## 4.2 Regimeskiftet 2022–2023 og dets konsekvenser

Lakseprisen steg kraftig fra høsten 2022 til sommeren 2023 — fra et nivå rundt 70–80 NOK/kg til over 110–120 NOK/kg. Dette *regimeskiftet* har tre viktige konsekvenser for studien:

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

SARIMA-orden (1,1,1)(1,1,1)₅₂ ble valgt manuelt. Refit-sensitivitetsanalysen (tabell 4.1) sammenlikner walk-forward-MAE for ulike refit-frekvenser — og gir et overraskende resultat: `refit=∞` (aldri refit, Spor A sin tilnærming) er best på alle tre horisonter.

**Tabell 4.1 – Refit-sensitivitet SARIMA(1,1,1)(1,1,1)₅₂**

| Refit (uker) | h=4 MAE | h=8 MAE | h=12 MAE | Kjøretid |
|---:|---:|---:|---:|---:|
| 4 | 8,517 | 11,281 | 13,459 | ~25 min |
| 12 | 8,460 | 11,207 | 13,317 | ~12 min |
| 26 | 8,451 | 11,163 | 13,299 | ~7,5 min |
| **∞ (Spor A)** | **8,270** | **11,074** | **13,151** | **~3 min** |

Den intuitive forventningen er at hyppigere re-estimering gir bedre prediksjoner. Her er det motsatte tilfelle: re-estimering på data som inkluderer boomperioden 2022–2023 forringer parameterkvaliteten fordi modellen trekkes mot den ekstraordinære prisdynamikken og blir dårligere kalibrert for normalt markedsklima. Å holde parameterene faste fra pre-boom-trening er dermed mer robust. Dette er nok et uttrykk for det samme regimeskift-problemet som påvirker alle metoder i studien.

## 4.5 Ensemble-averaging og horisontsensitivitet

Ensemble-averagering (50/50 vekting) er gunstig på h = 4 og h = 8, men nøytralt/marginalt negativt på h = 12 fordi XGBoost+ES er markant svakere enn LightGBM+ES på lang horisont (15,31 vs. 13,24 MAE). Post-hoc optimale vekter (w_XGB = 0,2 på h = 12) bekrefter at LightGBM sin bedre generalisering på h = 12 kan utnyttes med asymmetrisk vekting.

Dette reiser spørsmålet om online ensemble-vekting — adaptiv justering av vektene etter hvert som nye data blir tilgjengelig — kunne gitt ytterligere forbedring. Med 104 test-ukentlige observasjoner er det ikke tilstrekkelig statistisk power til å validere dynamisk vekting uten risiko for overfitting.
