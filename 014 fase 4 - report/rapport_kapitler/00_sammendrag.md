# Sammendrag

Denne rapporten undersøker mulighetene for maskinlæringsbasert prognose av ukentlig eksportpris for fersk norsk laks (NOK/kg) på tidshorisonter på 4, 8 og 12 uker frem i tid. Datasettene kombinerer SSBs ukentlige eksportstatistikk med valutakurs (EUR/NOK og USD/NOK, Norges Bank) og FAO sin kvartalsvise prisindeks for akvakultur — totalt 44 forklaringsvariabler etter feature-engineering.

Ni modellvarianter ble trent og evaluert på en felles testperiode (siste 104 uker, kronologisk splitt). Modellene spenner fra en naiv referansemodell over statistiske tidsseriemodeller (SARIMA og SARIMAX med rullende startpunkt) til gradientøkende tremodeller (XGBoost og LightGBM) og ensemble-kombinasjoner.

Ingen enkeltmodell dominerer alle horisonter:

- **h = 4 uker:** SARIMA oppnår lavest MAE (8,27 NOK/kg, 9,5 % MAPE), tett fulgt av Ensemble (8,33 NOK/kg). Diebold-Mariano-testen bekrefter at differansen **ikke er statistisk signifikant** (p = 0,89).
- **h = 8 uker:** Ensemblet (XGBoost + LightGBM med early stopping) er numerisk best med 10,85 NOK/kg (12,9 % MAPE), men heller ikke her er forskjellen fra SARIMA (11,07) signifikant (p = 0,93).
- **h = 12 uker:** SARIMAX med naiv valutakursprognose er best med 13,15 NOK/kg (15,8 % MAPE), tett fulgt av LightGBM tunet (13,06). DM-testen finner ingen signifikant forskjell mellom de to (p = 0,98).

Alle toppmodeller slår den naive baselines på samtlige horisonter; forbedringen er størst på h = 4 (3 % lavere MAPE enn naiv).

To sentrale metodiske funn må understrekes: (1) Gauss-baserte 95 %-konfidensintervaller for SARIMA/SARIMAX underdekker konsekvent (~80 % faktisk vs. 95 % nominell), fordi treningsresidualene har fettede haler og uoppfanget sesongautokorrelasjon. (2) Ensemblet viser systematisk negativ bias (–2,2 til –2,9 NOK/kg) som kan spores til lakseprisboomet 2022–2023 — en regimeskiftperiode modellen ikke ble trent på å gjenkjenne.

Studien konkluderer med at statistiske og maskinlæringsbaserte tilnærminger presterer **statistisk likeverdig** på tvers av horisonter, og at valget mellom modellene i praksis bør styres av driftsegenskaper (tolkbarhet, kjøretid, vedlikehold) snarere enn marginale punktestimatforskjeller.
