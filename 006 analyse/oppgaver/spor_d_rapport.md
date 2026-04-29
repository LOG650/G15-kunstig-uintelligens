# Spor D – Rapport-skriving og formidling

Lim hele dette dokumentet inn som første melding til AI-en. Den får da nok kontekst til å jobbe selvstendig.

---

## Prosjektkontekst

Studentprosjekt LOG650 (Logistikk og kunstig intelligens) ved HiM. Målet er å predikere ukentlig eksportpris for fersk norsk laks (NOK/kg) på horisontene 4, 8 og 12 uker fram.

Spor A, B og C er ferdige med modellering og dataarbeid (mai 2026). Du skriver rapporten og lager rapport-kvalitets-figurer fra de eksisterende resultatene. **Du skal ikke trene nye modeller.**

## Hovedstatus du jobber med

- 9 modellvarianter er trent og evaluert: Naiv, SARIMA (rullende), SARIMAX, XGBoost (utuned, tunet, +ES), LightGBM (tunet, +ES), Ensemble.
- 44 features i analyseklart datasett (inkl. FAO-imputation og Spor C-tilleggsfeatures).
- Alle resultater dokumentert i `006 analyse/LES_MEG.md` med samlet resultattabell, CI-kalibrering, residualdiagnostikk og kjente forbehold.
- Et utkast til rapport finnes som `014 fase 4 - report/Utkast til forskningsrapport.docx`.

## Dine oppgaver (i prioritert rekkefølge)

1. **Les `006 analyse/LES_MEG.md` grundig.** Den inneholder den autoritative resultattabellen og oppsummering av alle funn. Ikke fluffer egne tall – referer alltid til CSV-filer i `006 analyse/resultater/`.
2. **Skriv rapport-prosa i markdown** under `014 fase 4 - report/rapport_kapitler/` (opprett mappa). Filer:
   - `00_sammendrag.md` – 200-300 ord
   - `01_innledning.md` – problemstilling, motivasjon, forskningsspørsmål
   - `02_metode.md` – data, features, modellvalg, evaluering
   - `03_resultater.md` – tabeller og hovedfunn (referer eksisterende CSV-er)
   - `04_diskusjon.md` – tolkning av funn, modellbegrensninger (CI-underdekning, ensemble-bias, restautokorrelasjon)
   - `05_konklusjon.md` – hva fungerte, hva fungerte ikke, anbefalinger
   - `99_referanser.md` – kildehenvisninger (FAO, SSB, Norges Bank, statsmodels, scikit-learn, xgboost, lightgbm)
3. **Lag rapport-kvalitets-figurer** som er bedre enn de eksplorative i `resultater/`:
   - Modell-sammenligningstabell som figur (heatmap eller bar chart)
   - Beste prognose per horisont vs. faktisk pris (ren versjon med få farger)
   - Bias-plot for ensemble (allerede i `residualplot_*`, men lag rapport-versjon)
   Lagre som `006 analyse/resultater/rapport_*.png` og `rapport_*.pdf` (PDF for innebygging i Word/LaTeX).
4. **Sammenstilling**: når studenten er klar, kopier markdown-tekstene inn i Word-dokumentet (manuelt) eller produsér én samlet `rapport_full.md` for innliming.

## Filer du eier (lag/endre fritt)

- `014 fase 4 - report/rapport_kapitler/*.md` – kapittelutkast (opprettes av deg)
- `014 fase 4 - report/rapport_full.md` – samlet versjon (valgfri)
- `006 analyse/resultater/rapport_*.png` og `rapport_*.pdf` – rapport-kvalitets-figurer

## Filer du IKKE skal røre

- All modellkode (`sarima_eksperiment.py`, `ml_eksperiment.py`, `ml_ensemble.py`, `ml_residualplot.py`)
- `baseline_modeller.ipynb` og `eksporter_baseline.py` (frosset)
- `004 data/` – dataarbeid er ferdig
- Eksisterende `resultater/`-filer (les bare; ikke overskriv)
- `Utkast til forskningsrapport.docx` – la studenten kopiere inn manuelt selv

## Ting du må diskutere ærlig i rapporten

LES_MEG dokumenterer flere modellbegrensninger som **må rapporteres åpent**, ikke skjules:

- **SARIMA-CI underdekker** (~80 % faktisk vs. 95 % nominell) på grunn av fat-tailed residualer.
- **Restautokorrelasjon ved sesonglag 52** (Ljung-Box forkastet) – ordensvalget (1,1,1)(1,1,1,52) er en svakhet.
- **Ensemble har systematisk negativ bias** (-2.2 til -2.9 NOK/kg) – modellen underpredikerer konsekvent.
- **Ingen modell vinner alle horisontene** – SARIMA på h=4, Ensemble på h=8, SARIMAX på h=12.

Disse er reelle funn, ikke svakheter ved arbeidet. En god rapport diskuterer dem og foreslår hvordan de kan adresseres (Spor E, F, G adresserer flere av dem).

## Slik oppdaterer du delt status

Når et kapittel er ferdig, oppdater `006 analyse/LES_MEG.md` med en «Rapport»-status under "Neste steg".

## Suksesskriterium

Alle 7 markdown-kapitler eksisterer med substans (ikke bare overskrifter), referer korrekte tall fra CSV-ene, diskuterer modellbegrensningene ærlig, og kan kopieres rett inn i et Word- eller LaTeX-dokument. Minst 3 rapport-kvalitets-figurer i `resultater/rapport_*`.
