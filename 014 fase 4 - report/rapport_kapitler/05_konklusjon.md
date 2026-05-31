# 5. Konklusjon

## 5.1 Hva fungerte

**Statistiske tidsseriemodeller på korte horisonter.** SARIMA med rullende startpunkt (8,27 NOK/kg, 9,5 % MAPE på h = 4) er overlegen maskinlæringsmodellene på den korteste horisonten. Den kombinerer enkel treningstid, tolkbare parametre og god lokal autokorrelasjonsutnyttelse. Tillegget av EUR/USD-kurs (SARIMAX) gir marginal gevinst på h = 12 (12,93 vs. 13,15 NOK/kg) uten å skade de kortere horisontene.

**Ensemble med early stopping på mellomhorisonter.** XGBoost + LightGBM med early stopping og 40+ features vinner h = 8-konkurransen (10,85 NOK/kg). Early stopping er kritisk: ren hyperparameter-tuning uten regularisering overfitter og presterer dårligere enn naiv på korte horisonter.

**FAO-imputation som feature.** Forward-fill av FAO-kvartalsverdier inn i ukentlig oppløsning tillater inkludering av det globale markedssignalet som ellers ville gitt 75 % manglende data. Modellene viser at FAO-informasjonen er nyttig på h = 4 og h = 8.

**SHAP-tolkning er konsistent med domeneforståelse.** At `pris_lag_1` og `pris_lag_2` dominerer korte horisonter, mens `volum_sum_52u` og sesongkomponenten (`uke_cos`) er viktigst på h = 12, er intuitivt forsvarlig: lakseprisen er autokorrelert kortsiktig, men markedsbalansen (tilbudsvolum) og sesongmønsteret bestemmer retningen over et kvartal.

## 5.2 Hva fungerte ikke

**Kvantilregresjon for usikkerhetskvantifisering.** LightGBM kvantilregresjon trengt på pre-2024 data gir 35–46 % dekning på testperioden — under halvparten av det nominelle 95 %-målet. Regimeskiftet 2022–2023 gjør treningsdistribusjonen fundamentalt urepresentativ for testperioden.

**Gauss-CI fra SARIMA/SARIMAX.** Alle CI-metoder underdekker (~79–82 %), både Gauss og bootstrap. Rotårsaken er en kombinasjon av fettede residualhaler og gjenværende sesongautokorrelasjon. Reliabel usikkerhetskvantifisering for flerstegs lakseprognoser er et uløst problem i dette studiet.

**CV-basert bias-estimering.** TimeSeriesSplit-kryssvalidering i en ikke-stasjonær prisserie med regimeskift gir villedende bias-estimater. Metoden bør unngås for post-hoc bias-korreksjon i denne konteksten.

## 5.3 Anbefalinger

For operativt bruk anbefales en horisontstyrt modellstrategi. Tabellen under skiller mellom den beste *analytisk observerte* ytelsen og en *forsiktig operasjonell* anbefaling:

| Horisont | Anbefalt modell (operasjonelt) | MAE | Merk |
|---|---|---:|---|
| h = 4 uker | SARIMA(1,1,1)(1,1,1)₅₂, rullende | 8,27 NOK/kg | — |
| h = 8 uker | Ensemble (XGB+ES + LGBM+ES, lik 50/50 vekting) | 10,85 NOK/kg | Se note |
| h = 12 uker | SARIMAX(1,1,1)(1,1,1)₅₂ + EUR/USD | 12,93 NOK/kg | — |

**Note h = 8:** Post-hoc grid search over testsettet gir optimal vekting w_XGB = 0,8 (MAE 10,77 NOK/kg), men disse vektene er tilpasset det *kjente* testsettet og er ikke direkte overførbare til fremtidig drift. En operasjonell implementering bør fastlegge vekter på en separat valideringsperiode som holdes utenfor den endelige evalueringen. Lik 50/50-vekting (10,85 NOK/kg) er den robuste anbefalingen.

**Logistikkimplikasjoner:** Et prognosesystem basert på disse modellene har konkrete anvendelser langs laksens verdikjede:

- **Slakteplanlegging (h = 4 uker):** En MAE på 8,27 NOK/kg (~9,5 % MAPE) gir tilstrekkelig presisjon til å time slaktevolumer mot forventet prisutvikling. Oppdrettere kan bruke h=4-prognosen til å avgjøre om det lønner seg å forskyve slakting 1–2 uker.
- **Kontraktsinngåelse og prissikring (h = 8–12 uker):** På h=8–12 er feilmarginen 11–13 NOK/kg (~13–16 % MAPE). Dette er for upresist til å styre enkeltkontrakter, men nyttig for å vurdere om spotpriseksponering bør reduseres via terminkontrakter (Fish Pool).
- **Kapasitetsplanlegging i logistikk (h = 12 uker):** Prognose for 12 uker kan brukes av transportører og pakketerier til kapasitetsreservasjon: høy forventet pris signaliserer høy etterspørsel og behov for ekstra kjølekapasitet.

**Forbehold for praktisk bruk:** Alle tall er fra en historisk testperiode (2022–2024) som inkluderer en uvanlig prisopphøyingsperiode. Modellenes ytelse i mer normale markeder kan avvike. Konfidensintervallene skal tolkes med varsomhet — de dekker statistisk sett ~80 % av utfallene, ikke 95 %.

## 5.4 Videre arbeid

1. **Regime-bevisst modellering:** En Markov-vekslende SARIMA eller online-lærende ensemble som oppdager og tilpasser seg regimeskift kan adressere bias-problemet.
2. **Konformal prediksjon:** Konformal prediction (Vovk et al., 2005) gir garantert dekningsfrekvens uten Gauss-antagelse og bør utforskes som erstatning for Gauss-CI.
3. **Lengre datahistorikk:** Perioden 2000–2009 (med andre prisregimer) kan gi modellene bedre generalisering mot fremtidige regimeskift.
4. **Eksogene signaler:** Futurespriser på laks (Fish Pool Index), fôrkostnader og smoltutsett er potensielle leading indicators som kan forbedre lange horisonter ytterligere.
