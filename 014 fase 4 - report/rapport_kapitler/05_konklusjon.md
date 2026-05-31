# 5. Konklusjon

## 5.1 Hovedfunn og svar på problemstilling

Denne studien har undersøkt hvilke modeller som gir lavest prediksjonsfeil for ukentlig laksepris på 4, 8 og 12 ukers sikt. Numerisk er det en horisontsensitiv rangering:

*   **På kort sikt (4 uker)** oppnår **SARIMA** lavest MAE (8,27 NOK/kg), men Diebold-Mariano-testen bekrefter at forskjellen fra Ensemble (8,33 NOK/kg) ikke er statistisk signifikant (p = 0,89).
*   **På mellomlang sikt (8 uker)** er **Ensemblet** numerisk best (MAE 10,85 NOK/kg), men heller ikke her er differansen fra de neste beste modellene statistisk signifikant.
*   **På lang sikt (12 uker)** er **LightGBM tunet** og **SARIMAX_naiv** tilnærmet likeverdige (13,06 vs. 13,15 NOK/kg, p = 0,98).

Den overordnede konklusjonen er at statistiske og maskinlæringsbaserte metoder presterer **statistisk likeverdig** på alle horisonter innenfor dette datagrunnlaget. Ingen enkeltmodell er statistisk overlegen. Dette nyanserer den opprinnelige anbefalingen om horisontstyrt modellvalg: mens slik styring er *numerisk* motivert, er den ikke statistisk nødvendig med 104 testobservasjoner.

En viktig metodisk innsikt er at SARIMAX med naiv random-walk-valutaprognose presterer tilnærmet identisk med SARIMAX med perfekt fremtidig valutainformasjon (oracle) — Oracle-fordelen er ikke signifikant (DM = –1,11, p = 0,27). Dette bekrefter random-walk-hypotesen for valutakurser (Meese & Rogoff, 1983) og demonstrerer at SARIMAX er fullt operativ uten kunnskap om fremtidig valutakurs.

## 5.2 Praktiske implikasjoner

Studien viser at en kombinert modellstrategi kan gi forbedring på inntil 3 % i MAPE sammenlignet med naive estimater, men at de beste modellene er statistisk likeverdige. For sjømatnæringen innebærer dette at modellvalg bør styres av:

1. **Tolkbarhet:** SARIMA/SARIMAX er lettere å forklare for beslutningstagere enn ensemble-ML.
2. **Robusthet mot overfitting:** Ensemble med early stopping er mer robust enn tunet XGBoost alene.
3. **Driftsegenskaper:** SARIMA med faste parametere er raskest (~3 min) og enklest å vedlikeholde.

**Operasjonell anbefaling per horisont:** SARIMA eller SARIMAX_naiv for h = 4 (MAE ~8,24–8,27 NOK/kg), Ensemble for h = 8 (MAE 10,85 NOK/kg; lavest RMSE), og LightGBM tunet for h = 12 (MAE 13,06 NOK/kg; lavest MAPE).

Det må understrekes at alle modeller underestimerer usikkerheten i perioder med store regimeskift, slik som prishoppet i 2022–2023. Operative brukere bør derfor tolke konfidensintervallene som veiledende og ta høyde for at ekstreme utfall forekommer oftere enn modellene forutser.

## 5.3 Begrensninger og videre arbeid

Den største begrensningen i studien er modellenes sårbarhet for strukturelle regimeskift som ikke finnes i treningsdataene. Videre arbeid bør fokusere på:

1.  **Regime-bevisst modellering:** Utforske modeller som automatisk kan oppdage og tilpasse seg skifter i markedsvolatilitet, for eksempel gjennom online learning eller adaptive ensemble-vekter.
2.  **Konformal prediksjon:** Implementere rammeverk for *Conformal Prediction*. Dette er en teknikk som gir garantert dekningsgrad for usikkerhetsintervaller uten å hvile på urealistiske Gauss-antagelser om normalfordelte residualer. Dette ville løst problemet med systematisk underdekking identifisert i denne studien.
3.  **Integrasjon av Fish Pool-futures:** Inkludere futures-priser fra laksebørsen Fish Pool som en ledende indikator (eksogen variabel). Siden disse prisene reflekterer markedets samlede forventninger frem i tid, vil de sannsynligvis kunne redusere bias i perioder med store prisskift.
4.  **Utvide testperioden:** Med et lengere testsett (f.eks. 3–5 år) ville Diebold-Mariano-testen ha tilstrekkelig statistisk kraft til å avgjøre om horisontstyrt modellvalg er statistisk motivert.
5.  **Flere datakilder:** Integrere mer detaljerte tilbudsdata som biomasseoversikter og fôrsalg for å styrke de lengre horisontene ytterligere.
