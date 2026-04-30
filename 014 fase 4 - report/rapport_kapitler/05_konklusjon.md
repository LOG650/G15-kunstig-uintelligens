# 5. Konklusjon

## 5.1 Hva fungerte

**Statistiske tidsseriemodeller på korte horisonter.** SARIMA med rullende startpunkt (8,27 NOK/kg, 9,5 % MAPE på h = 4) er overlegen maskinlæringsmodellene på den korteste horisonten. Den kombinerer enkel treningstid, tolkbare parametre og god lokal autokorrelasjonsutnyttelse. Tillegget av EUR/USD-kurs (SARIMAX) gir marginal gevinst på h = 12 (12,93 vs. 13,15 NOK/kg) uten å skade de kortere horisontene.

**Ensemble med early stopping på mellomhorisonter.** XGBoost + LightGBM med early stopping og 40+ features vinner h = 8-konkurransen (10,85 NOK/kg). Early stopping er kritisk: ren hyperparameter-tuning uten regularisering overfitter og presterer dårligere enn naiv på korte horisonter.

**FAO-imputation som feature.** Spor C sitt forward-fill av FAO-kvartalsverdier inn i ukentlig oppløsning tillater inclusion av den globale markedssignalet som ellers ville gitt 75 % manglende data. Modellene viser at FAO-informasjonen er nyttig på h = 4 og h = 8.

**SHAP-tolkning er konsistent med domeneforståelse.** At `pris_lag_1` og `pris_lag_2` dominerer korte horisonter, mens `volum_sum_52u` og sesongkomponenten (`uke_cos`) er viktigst på h = 12, er intuitivt forsvarlig: lakseprisen er autokorrelert kortsiktig, men markedsbalansen (tilbudsvolum) og sesongmønsteret bestemmer retningen over et kvartal.

## 5.2 Hva fungerte ikke

**Kvantilregresjon for usikkerhetskvantifisering.** LightGBM kvantilregresjon trengt på pre-2024 data gir 35–46 % dekning på testperioden — under halvparten av det nominelle 95 %-målet. Regimeskiftet 2022–2023 gjør treningsdistribusjonen fundamentalt urepresentativ for testperioden.

**Gauss-CI fra SARIMA/SARIMAX.** Alle CI-metoder underdekker (~79–82 %), både Gauss og bootstrap. Rotårsaken er en kombinasjon av fettede residualhaler og gjenværende sesongautokorrelasjon. Reliabel usikkerhetskvantifisering for flerstegs lakseprognoser er et uløst problem i dette studiet.

**CV-basert bias-estimering.** TimeSeriesSplit-kryssvalidering i en ikke-stasjonær prisserie med regimeskift gir villedende bias-estimater. Metoden bør unngås for post-hoc bias-korreksjon i denne konteksten.

## 5.3 Anbefalinger

For operativt bruk anbefales en horisontstyrt modellstrategi:

| Horisont | Anbefalt modell | MAE (testperiode) |
|---|---|---:|
| h = 4 uker | SARIMA(1,1,1)(1,1,1)₅₂, rullende | 8,27 NOK/kg |
| h = 8 uker | Ensemble (XGB+ES + LGBM+ES, w=0,8 XGB) | 10,77 NOK/kg* |
| h = 12 uker | SARIMAX(1,1,1)(1,1,1)₅₂ + EUR/USD | 12,93 NOK/kg |

*Med post-hoc optimal vekting.

**Forbehold for praktisk bruk:** Alle tall er fra en historisk testperiode (2022–2024) som inkluderer en uvanlig prisopphøyingsperiode. Modellenes ytelse i mer normale markeder kan avvike. Konfidensintervallene skal tolkes med varsomhet — de dekker statistisk sett ~80 % av utfallene, ikke 95 %.

## 5.4 Videre arbeid

1. **Regime-bevisst modellering:** En Markov-vekslende SARIMA eller online-lærende ensemble som oppdager og tilpasser seg regimeskift kan adressere bias-problemet.
2. **Konformal prediksjon:** Konformal prediction (Vovk et al., 2005) gir garantert dekningsfrekvens uten Gauss-antagelse og bør utforskes som erstatning for Gauss-CI.
3. **Lengre datahistorikk:** Perioden 2000–2009 (med andre prisregimer) kan gi modellene bedre generalisering mot fremtidige regimeskift.
4. **Eksogene signaler:** Futurespriser på laks (Fish Pool Index), fôrkostnader og smoltutsett er potensielle leading indicators som kan forbedre lange horisonter ytterligere.
