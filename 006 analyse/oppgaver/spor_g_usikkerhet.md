# Spor G – Usikkerhetskvantifisering (empiriske CI for SARIMA og ML)

Lim hele dette dokumentet inn som første melding til AI-en. Den får da nok kontekst til å jobbe selvstendig.

---

## Prosjektkontekst

Studentprosjekt LOG650 (Logistikk og kunstig intelligens) ved HiM. Mål: predikere ukentlig eksportpris for fersk norsk laks (NOK/kg) på horisontene 4, 8, 12 uker.

Spor A's SARIMA-CIer er **systematisk for smale**: empirisk dekning ~80 % vs. 95 % nominell, fordi de er Gauss-baserte og residualene har fat tails. Spor B's ML-prediksjoner har overhodet ingen usikkerhetsestimater.

Du skal lage **empirisk kalibrerte intervaller** for både SARIMA og ML-modellen, og dokumentere hvor mye bedre de er enn dagens. Du **skal ikke** endre Spor A eller Spor B's eksisterende skript.

## Status og baseline du må slå (fra `006 analyse/resultater/sarima_ci_dekning.csv`)

| Modell | h=4 dekning | h=8 dekning | h=12 dekning |
|---|---:|---:|---:|
| SARIMA (Gauss) | 79 % | 80 % | 81 % |
| SARIMAX (Gauss) | 81 % | 81 % | 82 % |

Mål: empirisk dekning skal være innenfor 95 % ± 3 prosentpoeng (dvs. 92 – 98 %) på testsettet, uten at intervallene blir absurd brede.

ML-modellene har ingen CI per i dag. Du skal lage dem.

## Dine oppgaver (i prioritert rekkefølge)

1. **Bootstrap-CI for SARIMA og SARIMAX**. Bruk in-sample residualer fra `init.standardized_forecasts_error[0]` etter slipping av første 52 obs (for å unngå init-effekter). Sample residualer med erstatning, addér til punktprognose, og dann empiriske 2.5/97.5-percentiler. Dette gir CI som respekterer fat tails og skjevhet i residualfordelingen. Sammenlign empirisk dekning på testsettet mot dagens Gauss-CI.
2. **Quantile regression med LightGBM** for ensemble-prediksjonen. Tren én LightGBM-modell per (horisont, kvantil) for q ∈ {0.025, 0.5, 0.975}, samme features som Spor B brukte i ml_ensemble. Bruk `LGBMRegressor(objective="quantile", alpha=q)`. Lag prediktive intervaller fra q=0.025 og q=0.975. Mål empirisk dekning. Sentrum (q=0.5) bør være nært ensemble-mean for sanity-check.
3. **Kalibreringssammenligning**. Lag plot av empirisk dekning vs. nominell for alle metoder, alle horisonter (subplot per horisont). En perfekt kalibrert metode ligger på y=x-linjen.
4. **Reliability/sharpness-tradeoff**. Smale intervaller er bra hvis de er kalibrerte. Rapportér gjennomsnittlig bredde av CI-ene per metode/horisont – hvilken metode gir best dekning til lavest pris i bredde?
5. **Stretch (nice-to-have)**: konformal prediksjon (`mapie` eller manuell implementasjon) som teoretisk garantert kalibrert metode.

## Filer du eier (lag/endre fritt)

- `006 analyse/usikkerhet_eksperiment.py` – hovedskript (kan opprettes)
- `006 analyse/resultater/usikkerhet_sarima_bootstrap.csv` – kolonner `modell, horisont, n, dekning, gj_bredde`
- `006 analyse/resultater/usikkerhet_ml_quantile.csv` – samme kolonner for ML
- `006 analyse/resultater/usikkerhet_kalibrering.csv` – samlet sammenligning Gauss vs. bootstrap vs. quantile
- `006 analyse/resultater/usikkerhet_kalibrering.png` – reliability-plot
- `006 analyse/resultater/usikkerhet_sharpness.png` – bredde vs. dekning per metode

## Filer du IKKE skal røre

- `006 analyse/sarima_eksperiment.py`, `ml_*.py` – Spor A og B's skript er frosset (les bare)
- `006 analyse/resultater/sarima_*.csv`, `ml_*.csv` (eksisterende) – les bare
- `004 data/` – dataarbeid er ferdig

## Hva du leser fra Spor A og B

- `006 analyse/sarima_eksperiment.py`: oppskrift på rullende-opphav-evaluering (kopiér rullerende_prognose-funksjon hvis du trenger).
- `006 analyse/resultater/sarima_prognose_h{4,8,12}.csv`, `sarimax_prognose_h{4,8,12}.csv`: dagens punktprognoser og Gauss-CI for sammenligning.
- `006 analyse/ml_ensemble.py`: hvordan ensemblet trenes med early stopping. Du må trene quantile-versjoner på lignende måte.
- `006 analyse/resultater/ml_ensemble_prediksjoner.csv`: testsett-prediksjoner for ensemble (faktisk vs. predikert).

## Slik oppdaterer du delt status

Når du er ferdig, oppdatér `006 analyse/LES_MEG.md`:
- Legg til en seksjon "Kalibrerte intervaller" eller utvid CI-kalibrerings-tabellen.
- Stryk problem-punkt 4 (SARIMA-CI underdekker) hvis bootstrap løser det – ellers nyansér.
- Stryk neste-steg-punkt e (empirisk kalibrerte intervaller).
- Oppdater "Sist oppdatert"-datoen.

## Suksesskriterium

`usikkerhet_sarima_bootstrap.csv` viser dekning innenfor 92–98 % på minst 2 av 3 horisonter. `usikkerhet_ml_quantile.csv` finnes med kalibreringstall og rimelige intervallbredder. Reliability-plot viser at minst én ny metode er klart bedre kalibrert enn dagens Gauss-CI.
