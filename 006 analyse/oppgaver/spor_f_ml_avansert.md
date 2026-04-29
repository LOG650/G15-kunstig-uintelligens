# Spor F – ML-utvidelser (bias-korreksjon, ensemble-vekting, SHAP)

Lim hele dette dokumentet inn som første melding til AI-en. Den får da nok kontekst til å jobbe selvstendig.

---

## Prosjektkontekst

Studentprosjekt LOG650 (Logistikk og kunstig intelligens) ved HiM. Mål: predikere ukentlig eksportpris for fersk norsk laks (NOK/kg) på horisontene 4, 8, 12 uker. Spor B trente XGBoost (utuned, tunet, +ES), LightGBM (tunet, +ES), og et uvektet ensemble; resultater i `006 analyse/resultater/`.

Du skal **forbedre Spor B's modeller** med tre konkrete justeringer. Du **skal ikke** endre Spor B's eksisterende skript.

## Status og baseline du må slå (fra `006 analyse/resultater/ml_ensemble.csv`)

| Modell | h=4 MAE | h=8 MAE | h=12 MAE |
|---|---:|---:|---:|
| XGBoost+ES | 8.71 | 10.88 | 15.31 |
| LightGBM+ES | 8.85 | 11.53 | **13.24** |
| **Ensemble** (uvektet snitt) | **8.33** | **10.85** | 13.56 |

Residualanalyse (`ml_residualar.csv`) avdekket **systematisk negativ bias** på -2.16 / -2.92 / -2.73 NOK/kg per horisont – ensemblet underpredikerer konsekvent. På h=12 vinner LightGBM+ES alene over ensemblet, dvs. uvektet snitt er ikke optimalt.

## Dine oppgaver (i prioritert rekkefølge)

1. **Bias-korreksjon**. Estimer biasen `b_h` for hvert horisont fra trenings-CV-residualer (ikke testresidualer – det ville være lekkasje). Bruk samme `TimeSeriesSplit(n_splits=5)` som Spor B brukte for tuning. Lag `Ensemble_bias_korr` = `Ensemble + b_h`. Sammenlign MAE før/etter, og dekompondér feilen i bias og varians.
2. **Optimal ensemble-vekting**. For hvert horisont, finn beste `w` i `w*XGBoost+ES + (1-w)*LightGBM+ES` via grid (`w ∈ {0.0, 0.1, ..., 1.0}`) på trenings-CV-residualer. Sammenlign vektet ensemble mot uvektet på testsettet. Forventning: w=0 (kun LightGBM) på h=12, w≈0.5 ellers.
3. **SHAP-verdier** for vinnermodellen per horisont (eller for LightGBM hvis enkel beste). Bruk `shap.TreeExplainer`. Lagre top-10 viktigste features og en SHAP-summary-plot. Dette gir tolkbarhet til rapporten.
4. **Stretch (nice-to-have)**: prøve `IsotonicRegression` på treningsresidualer for å lære en ikke-lineær kalibrering, ikke bare en konstant bias.

## Filer du eier (lag/endre fritt)

- `006 analyse/ml_avansert.py` – hovedskript (kan opprettes)
- `006 analyse/resultater/ml_avansert_bias_korr.csv` – kolonner `horisont, modell, MAE_for, MAE_etter, bias_korreksjon`
- `006 analyse/resultater/ml_avansert_vekter.csv` – kolonner `horisont, beste_w, MAE_uvektet, MAE_vektet`
- `006 analyse/resultater/ml_avansert_shap_h{4,8,12}.csv` – top features per horisont
- `006 analyse/resultater/ml_avansert_shap_h{4,8,12}.png` – SHAP-summary-plot

## Filer du IKKE skal røre

- `006 analyse/ml_eksperiment.py`, `ml_ensemble.py`, `ml_residualplot.py` – Spor B's skript er frosset (les bare)
- `006 analyse/resultater/ml_*.csv` (eksisterende) – les bare
- `sarima_*` – tilhører Spor E
- `usikkerhet_*` – tilhører Spor G
- `004 data/` – dataarbeid er ferdig

## Hva du leser fra Spor B

- `006 analyse/ml_eksperiment.py` og `ml_ensemble.py` for hvordan modellene er trent og evaluert (kopiér hjelpekode hvis du trenger det – ikke endre originalen).
- `006 analyse/resultater/ml_ensemble_prediksjoner.csv` har faktisk og predikert pris per uke/horisont/modell – dette er nyttig for vekting og bias-analyse uten å trene på nytt.
- `006 analyse/resultater/ml_residualar.csv` har residualer per uke/horisont.

For SHAP og bias-estimering trenger du å trene modellen på nytt (med samme params som Spor B), eller lese inn de tunede modellene hvis de er lagret som `.pkl` (sjekk i Spor B's resultater).

## Slik oppdaterer du delt status

Når du er ferdig, oppdatér `006 analyse/LES_MEG.md`:
- Legg til rader for `Ensemble_bias_korr` og `Ensemble_vektet` i resultattabellen.
- Oppdatér residualanalyse-seksjonen med MAE etter bias-korreksjon.
- Stryk problem-punkt 6 (ensemble-bias) hvis bias-korreksjon løser det.
- Stryk neste-steg-punkt d (bias-korreksjon) og f (ensemble-vekting).
- Oppdater "Sist oppdatert"-datoen.

## Suksesskriterium

`ml_avansert_bias_korr.csv`, `ml_avansert_vekter.csv` og `ml_avansert_shap_h*.csv` finnes. Klar konklusjon i LES_MEG: gir bias-korreksjon en MAE-forbedring? Hva er optimal vekting per horisont? Top-3 viktigste features per horisont rapportert.
