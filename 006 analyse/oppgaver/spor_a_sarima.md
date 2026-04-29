# Spor A – SARIMA-sporet

Lim hele dette dokumentet inn som første melding til AI-en. Den får da nok kontekst til å jobbe selvstendig.

---

## Prosjektkontekst

Studentprosjekt LOG650 (Logistikk og kunstig intelligens) ved HiM. Vi skal predikere ukentlig eksportpris for fersk norsk laks (NOK/kg) på horisontene 4, 8 og 12 uker fram. Tre studenter jobber parallelt; du jobber kun på SARIMA-sporet.

Arbeidskatalog: `006 analyse/`. Datasettet ligger i `004 data/Analyseklart datasett/laks_ukentlig_features.csv` og er beskrevet i `004 data/Analyseklart datasett/LES_MEG.md`. Periode: 2010-01-04 til 2026-03-09 (845 ukentlige observasjoner). Testset = siste 104 uker. Mål: MAE og MAPE per horisont.

## Baseline du må slå (fra `006 analyse/resultater/baseline_metrikker_pivot.csv`)

| Horisont | Naiv MAE | SARIMA MAE (urettferdig) | XGBoost MAE |
|---|---|---|---|
| 4 | 8.51 | 22.19 | 11.46 |
| 8 | 13.04 | 22.19 | 12.59 |
| 12 | 16.35 | 22.19 | 14.79 |

SARIMA-tallene er identiske fordi nåværende implementering gjør single-fit + multi-step forecast – samme prognoseserie sammenlignes for alle horisonter. Det er nettopp dette du skal fikse.

## Dine oppgaver (i prioritert rekkefølge)

1. **Rullende-opphav-evaluering for SARIMA(1,1,1)(1,1,1,52).** For hver uke `t` i testperioden: refit eller `extend` modellen på data til og med `t-h`, hent ut prediksjonen `h` skritt fram, sammenlign med faktisk pris på `t`. Gjør dette for h ∈ {4, 8, 12}. Rapportér MAE og MAPE per horisont.
2. **SARIMAX med eksogene variabler.** Legg til `eur_nok_snitt` og `usd_nok_snitt` som eksogene regressorer. Sammenlign mot ren SARIMA. Bruk samme rullende-opphav-evaluering.
3. **Konfidensintervaller (95 %)** for prognosene fra punkt 1 og 2. Plot dem mot fasit på testperioden.

Refit per uke kan være tregt – bruk `results.append(new_obs, refit=False)` der det går, eller refit hver 4./12. uke hvis tiden blir uoverkommelig. Dokumentér valget.

## Filer du eier (lag/endre fritt)

- `006 analyse/sarima_eksperiment.py` – hovedskript (kan opprettes)
- `006 analyse/sarima_arbeid.ipynb` – eksplorativ notebook hvis du vil (kan opprettes)
- `006 analyse/resultater/sarima_metrikker.csv` – output (skriv long-format: kolonner `modell, horisont, n, MAE, MAPE`)
- `006 analyse/resultater/sarima_*.png` – evt. figurer

## Filer du IKKE skal røre

- `006 analyse/baseline_modeller.ipynb` – frosset som referanse
- `006 analyse/eksporter_baseline.py` – frosset
- `006 analyse/ml_*.py`, `xgboost_*.py`, `lgbm_*.py` – tilhører Spor B
- `004 data/` og alle filer der – tilhører Spor C
- Eksisterende kolonner i features-CSV kan brukes, men IKKE bygg datasettet på nytt

## Slik oppdaterer du delt status

Når du er ferdig med en oppgave, oppdatér `006 analyse/LES_MEG.md`:
- Bytt ut SARIMA-radene i resultattabellen med dine nye, korrekte tall
- Stryk det første punktet under "Kjente problemer" (SARIMA-evaluering)
- Stryk relevante punkt under "Neste steg"
- Oppdatér "Sist oppdatert"-datoen

Hvis du får merge-konflikt på LES_MEG.md, behold begge endringene – tabellen og listene er additive.

## Suksesskriterium

`sarima_metrikker.csv` finnes med tre rader (en per horisont) der MAE varierer med h, ikke er identisk. Helst slår SARIMAX naiv på minst én horisont.
