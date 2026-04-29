# Spor E – SARIMA-utvidelser (auto-ARIMA, refit-sensitivitet)

Lim hele dette dokumentet inn som første melding til AI-en. Den får da nok kontekst til å jobbe selvstendig.

---

## Prosjektkontekst

Studentprosjekt LOG650 (Logistikk og kunstig intelligens) ved HiM. Mål: predikere ukentlig eksportpris for fersk norsk laks (NOK/kg) på horisontene 4, 8, 12 uker. Spor A trente SARIMA(1,1,1)(1,1,1,52) og SARIMAX med EUR/USD som eksogene; resultater i `006 analyse/resultater/sarima_metrikker.csv`.

Du skal **validere SARIMA-valget** og dokumentere robustheten til Spor A's hurtigløsning. Du **skal ikke** endre Spor A's eksisterende skript.

## Status og baseline du må slå (fra `006 analyse/resultater/sarima_metrikker.csv`)

| Modell | h=4 MAE | h=8 MAE | h=12 MAE |
|---|---:|---:|---:|
| SARIMA(1,1,1)(1,1,1,52) rullende, refit=False | 8.27 | 11.07 | 13.15 |
| SARIMAX (samme + EUR/USD eksogen), refit=False | 8.33 | 11.07 | 12.93 |

Residualdiagnostikk (Spor A) avdekket **restautokorrelasjon ved sesonglag 52** for begge modeller (Ljung-Box p < 0.001). Det er motivasjonen for auto-ARIMA-eksperimentet.

## Dine oppgaver (i prioritert rekkefølge)

1. **Auto-ARIMA-verifikasjon** med `pmdarima.auto_arima`. Kjør med `seasonal=True, m=52, max_p=3, max_q=3, max_P=2, max_Q=2, max_d=2, max_D=1, stepwise=True, information_criterion='aic'`. Forventet kjøretid: 10–30 minutter. Sammenlign foreslått ordre med (1,1,1)(1,1,1,52). Rapportér AIC, BIC, og residualdiagnostikk for begge.
2. **Rullende-opphav-evaluering av auto-ARIMA-ordren**. Bruk samme oppskrift som Spor A (`append(refit=False)`). Sammenlign MAE/MAPE per horisont mot Spor A's tall.
3. **Refit-sensitivitetsanalyse**. Kjør Spor A's logikk med `refit_hver_uke ∈ {1, 4, 12, 26, ∞}` (∞ = aldri, dvs. dagens refit=False). Rapportér MAE per horisont og kjøretid. Validerer eller utfordrer Spor A's hurtigløsning.
4. **Stretch (nice-to-have)**: Holt-Winters seasonal (`statsmodels.tsa.holtwinters.ExponentialSmoothing`) som alternativ statistisk benchmark. Eller STL-dekomposisjon + ARIMA på residual-komponenten.

## Filer du eier (lag/endre fritt)

- `006 analyse/sarima_avansert.py` – hovedskript (kan opprettes)
- `006 analyse/resultater/sarima_avansert_autoarima.csv` – auto-ARIMA-ordre, AIC/BIC, MAE/MAPE
- `006 analyse/resultater/sarima_avansert_refit_sensitivitet.csv` – kolonner `refit_hver_uke, horisont, MAE, MAPE, sek_kjoretid`
- `006 analyse/resultater/sarima_avansert_*.png` – plot

## Filer du IKKE skal røre

- `006 analyse/sarima_eksperiment.py` – Spor A's resultater er frosset (les bare)
- `006 analyse/resultater/sarima_*.csv` (eksisterende) – les bare
- `ml_*` – tilhører Spor F
- `usikkerhet_*` – tilhører Spor G
- `004 data/` – dataarbeid er ferdig

## Hva du leser fra Spor A

- `006 analyse/sarima_eksperiment.py` for å se nøyaktig hvordan rullende opphav er implementert (kopiér hjelpefunksjoner som `evaluer`, `rullerende_prognose` til ditt eget skript hvis du trenger dem).
- `006 analyse/resultater/sarima_metrikker.csv` for sammenligningstall.

## Slik oppdaterer du delt status

Når du er ferdig, oppdater `006 analyse/LES_MEG.md`:
- Legg til en seksjon eller rader i resultattabellen for auto-ARIMA-modellen og refit-sensitivitet.
- Hvis auto-ARIMA gir bedre eller verre orden, oppdater problem-listen punkt 5 (restautokorrelasjon ved sesonglag).
- Stryk eller marker punkt b og c under "Neste steg" som ferdig.
- Oppdatér "Sist oppdatert"-datoen.

## Suksesskriterium

`sarima_avansert_autoarima.csv` finnes med foreslått ordre + tilhørende AIC/BIC/MAE. `sarima_avansert_refit_sensitivitet.csv` har 5 rader (refit-frekvenser) × 3 horisonter med MAE og kjøretid. Klar konklusjon i LES_MEG om hvorvidt (a) auto-ARIMA finner en bedre orden, og (b) refit=False er en god tilnærming.
