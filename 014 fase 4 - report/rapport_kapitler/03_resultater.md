# 3. Resultater

## 3.1 Prediksjonsnøyaktighet

Tabell 3.1 viser MAE og MAPE for alle modeller på testperioden (104 uker). Beste modell per horisont er uthevet.

**Tabell 3.1 – Prognoseytelse på testsettet (siste 104 uker)**

| Modell | h=4 MAE | h=4 MAPE | h=8 MAE | h=8 MAPE | h=12 MAE | h=12 MAPE |
|---|---:|---:|---:|---:|---:|---:|
| Naiv (`pris(t-h)`) | 8,51 | 9,8 % | 13,04 | 15,4 % | 16,35 | 19,7 % |
| **SARIMA** (rullende) | **8,27** | **9,5 %** | 11,07 | 13,1 % | 13,15 | 15,9 % |
| **SARIMAX** (rullende, EUR/USD) | 8,33 | 9,6 % | 11,07 | 13,1 % | **12,93** | **15,6 %** |
| XGBoost (utunet, baseline) | 11,46 | 13,7 % | 12,59 | 15,1 % | 14,79 | 18,1 % |
| XGBoost (tunet) | 10,37 | 12,2 % | 11,98 | 14,3 % | 15,47 | 19,0 % |
| LightGBM (tunet) | 10,45 | 12,3 % | 10,90 | 12,9 % | 13,06 | 16,0 % |
| XGBoost + early stopping | 8,71 | 10,1 % | 10,88 | 12,9 % | 15,31 | 18,7 % |
| LightGBM + early stopping | 8,85 | 10,4 % | 11,53 | 13,7 % | 13,24 | 16,3 % |
| **Ensemble** (XGB+ES + LGBM+ES) | 8,33 | 9,6 % | **10,85** | **12,9 %** | 13,56 | 16,7 % |

*Kilde: `resultater/sarima_metrikker.csv`, `resultater/ml_ensemble.csv`.*

**Viktige observasjoner:**

- Alle de tre toppmodellene (SARIMA, SARIMAX, Ensemble) slår naiv referansen på samtlige horisonter.
- SARIMA og ensemblet er statistisk likt på h = 4 (8,27 vs. 8,33 NOK/kg, differanse 0,06 NOK/kg).
- Ensemblet vinner h = 8 med kun 0,03 NOK/kg over XGBoost+ES — marginalt, men konsistent.
- På h = 12 dominerer SARIMAX med 12,93 NOK/kg mot LightGBM tunet (13,06) og LightGBM+ES (13,24). Ensemble-averaging skader her fordi XGBoost+ES er svakere (15,31).
- XGBoost-tuning alene (uten early stopping) er svakere enn naiv på h = 4 og h = 8 — hyperparameter-tuning uten regularisering overfitter.

## 3.2 Kalibrering av konfidensintervaller

Gauss-baserte 95 %-konfidensintervaller fra SARIMA/SARIMAX underdekker systematisk. Tabell 3.2 viser empirisk dekning og gjennomsnittlig bredde.

**Tabell 3.2 – CI-kalibrering (nominelt 95 %)**

| Metode | h=4 dekning | h=8 dekning | h=12 dekning | Gj. bredde h=4 (NOK/kg) |
|---|---:|---:|---:|---:|
| SARIMA Gauss | 79,2 % | 80,4 % | 80,6 % | 26,1 |
| SARIMAX Gauss | 81,2 % | 81,4 % | 81,7 % | 26,2 |
| SARIMA bootstrap | 80,2 % | 79,4 % | 80,6 % | 25,8 |
| SARIMAX bootstrap | 76,2 % | 79,4 % | 78,5 % | 24,3 |
| LightGBM kvantilregresjon | 46,2 % | 46,2 % | 34,6 % | 18,2 |

*Kilde: `resultater/sarima_ci_dekning.csv`, `resultater/usikkerhet_kalibrering.csv`.*

Ingen metode når det nominelle 95 %-målet. Bootstrap-tilnærmingen reproduserer omtrent Gauss-dekning (~79–81 %) etter residualskalering, men tillegger ikke ny verdi. LightGBM kvantilregresjon underdekker kraftig (35–46 %) på grunn av regimeskiftet 2022–2023 (se seksjon 4.2).

## 3.3 Residualdiagnostikk (SARIMA / SARIMAX)

Tabell 3.3 oppsummerer statistiske tester på in-sample treningsresidualene (689 observasjoner).

**Tabell 3.3 – Residualtester (treningssett)**

| Modell | LB(10) p | LB(20) p | LB(52) p | Skjevhet | Kurtose |
|---|---:|---:|---:|---:|---:|
| SARIMA  | 0,004 | 0,002 | < 0,001 | +0,24 | 4,55 |
| SARIMAX | 0,009 | 0,004 | < 0,001 | +0,28 | 4,44 |

*Kilde: `resultater/sarima_residualdiagnostikk.csv`.*

Ljung-Box-testen forkaster hvit-støy-hypotesen på alle lag (p < 0,01), med særlig sterk effekt ved lag 52. Dette indikerer gjenværende sesongautokorrelasjon som (1,1,1)(1,1,1)₅₂-ordenen ikke fanger fullt ut. Kurtose ≈ 4,5 bekrefter fettede haler relativt til normalfordelingen — den direkte årsaken til CI-underdekking.

## 3.4 Bias-korreksjon og ensemble-vekting (Spor F)

**Tabell 3.4 – Bias-korreksjon på ensemble-prediksjoner**

| Horisont | Kjent bias (NOK/kg) | MAE før | MAE etter | Endring |
|---:|---:|---:|---:|---:|
| 4 | −2,16 | 8,33 | **8,11** | −0,21 |
| 8 | −2,92 | 10,85 | **10,60** | −0,25 |
| 12 | −2,73 | 13,56 | 13,71 | +0,15 |

Bias-korreksjon hjelper på h = 4 og h = 8, men øker MAE marginalt på h = 12 fordi feilene der er mer symmetrisk fordelt.

**Tabell 3.5 – Optimal ensemble-vekting (w = andel XGBoost)**

| Horisont | Beste w_XGB | MAE uvektet | MAE vektet | Endring |
|---:|---:|---:|---:|---:|
| 4 | 0,5 | 8,33 | 8,33 | 0,00 |
| 8 | 0,8 | 10,85 | **10,77** | −0,08 |
| 12 | 0,2 | 13,56 | **13,22** | −0,34 |

*Kilde: `resultater/ml_avansert_bias_korr.csv`, `resultater/ml_avansert_vekter.csv`.*

På h = 12 dominerer LightGBM (w_XGB = 0,2 gir 80 % LightGBM-vekting), noe som bekrefter funnene fra enkeltmodell-sammenligningen.

## 3.5 Feature-viktighet (SHAP)

Tabell 3.6 viser de tre viktigste featurene per horisont fra LightGBM SHAP-analyse (gjennomsnittlig absolutt SHAP-verdi).

**Tabell 3.6 – Top-3 features per horisont (LightGBM SHAP)**

| Horisont | Rang 1 | Rang 2 | Rang 3 |
|---:|---|---|---|
| 4 | `pris_lag_1` | `pris_lag_2` | `pris_ma_4` |
| 8 | `pris_lag_1` | `pris_ma_4` | `uke_cos` |
| 12 | `volum_sum_52u` | `uke_cos` | `pris_ma_4` |

*Kilde: `resultater/ml_avansert_shap_h4.csv`, `ml_avansert_shap_h8.csv`, `ml_avansert_shap_h12.csv`.*

Lagfeaturer (`pris_lag_1`, `pris_lag_2`) dominerer korte horisonter, i tråd med at lakseprisen viser sterk korttidsautokorrelasjon. På h = 12 overtar eksportvolum på 52-ukersvindu (`volum_sum_52u`) og det sesongmessige cosinussignalet (`uke_cos`) — reflekterer at strukturelle sesong- og markedsbalanseforhold er viktigere for langsiktige prognoser.
