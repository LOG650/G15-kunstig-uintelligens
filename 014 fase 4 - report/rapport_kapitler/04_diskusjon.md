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

1. **Fettede haler (kurtose ≈ 4,5):** Gauss-antagelsen underestimerer sannsynligheten for store avvik. De 2,5 % og 97,5 % kvantilene i treningsresidualene er smalere enn ±1,96σ tilsier.

2. **Uoppfanget sesongautokorrelasjon:** Ljung-Box-testene forkaster hvit-støy-hypotesen kraftig ved lag 52 (p ≪ 0,001). Modellen klarer ikke å absorbere all sesongstruktur i (1,1,1)(1,1,1)₅₂-ordenen. De gjenværende korrelerte residualene blåser opp den sanne usikkerheten utover hva Gauss-CI fanget.

Bootstrap-skalering (seksjon 3.2) reproduserte omtrent samme dekning som Gauss, men uten forbedring. Dette skyldes at bootstrapen sampler fra de *samme* in-sample residualene — dermed arves de samme egenskapene (fettede haler og sesongkorrelasjon) inn i intervallestimatet.

En reell forbedring av CI-kalibrerringen krever enten (a) en rikere residualmodell (t-fordeling, ikke-parametrisk), (b) conformal prediction-rammeverk, eller (c) å trekke residualene fra en lengere kalibrerings-periode som representerer fremtidsregimet.

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
- **h = 12:** LightGBM+ES (RMSE 16,93) slår LightGBM tunet og SARIMAX_naiv på RMSE.

For operativ logistikk er valget mellom MAE og RMSE knyttet til konsekvensene av store feil. Dersom en stor enkeltfeil (f.eks. en prisbevegelse på 20 NOK/kg som modellen ikke fanget) medfører uforholdsmessig store kostnader (tapte kontrakter, feilallokert kapasitet), er RMSE-minimering å foretrekke. Dersom kostnadene er proporsjonale med avvikets størrelse (lineær tap), er MAE riktig metrikk. For slakteplanlegging og logistikkbooking er en rimelig antagelse at kostnadene er tilnærmet lineære, noe som støtter MAE som primærmetrikk. For prissikringsapplikasjoner (opsjoner, futures) vil imidlertid store avvik typisk medføre asymmetriske kostnader, og RMSE-vinneren bør foretrekkes.

## 4.7 Implikasjoner for logistikk og beslutningsstøtte

Resultatene fra studien har flere praktiske implikasjoner for aktører i sjømatnæringen. Mer presise prisprognoser, selv med de usikkerhetene som er identifisert, gir et bedre grunnlag for operativ beslutningsstøtte på flere områder:

1.  **Produksjons- og slakteplanlegging:** Ved å ha en indikasjon på prisutviklingen 4–8 uker frem i tid, kan oppdrettere i større grad optimalisere slaktetidspunktet. Dersom modellen indikerer et prisfall, kan det være lønnsomt å fremskynde slakting, og vice versa.
2.  **Eksportstrategi og prissikring:** For eksportører gir prognosene på 8 og 12 uker (hvor henholdsvis ensemblet og SARIMAX_naiv/LightGBM presterer best) et verktøy for å vurdere risiko i spotmarkedet opp mot faste kontrakter.
3.  **Logistikk og kapasitetsutnyttelse:** Bedre prisprognoser henger ofte sammen med forventet markedsetterspørsel. Dette muliggjør mer effektiv planlegging av transportkapasitet og logistikkflyt ut til de globale markedene.
4.  **Risikostyring:** Selv om konfidensintervallene underdekker (80 % mot 95 %), gir de en kvantifiserbar ramme for "worst-case" scenarier som er mer presis enn ren intuisjon. Forbedrede prognoser bidrar dermed til å redusere den økonomiske risikoen knyttet til den iboende prisvolatiliteten i sektoren.
