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
