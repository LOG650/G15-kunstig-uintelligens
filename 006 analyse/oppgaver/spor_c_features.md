# Spor C – Feature engineering / data-sporet

Lim hele dette dokumentet inn som første melding til AI-en. Den får da nok kontekst til å jobbe selvstendig.

---

## Prosjektkontekst

Studentprosjekt LOG650 (Logistikk og kunstig intelligens) ved HiM. Vi skal predikere ukentlig eksportpris for fersk norsk laks (NOK/kg) på horisontene 4, 8 og 12 uker fram. Tre studenter jobber parallelt; du jobber kun på data-/feature-sporet.

Datasettet bygges fra rådata i `004 data/` og skrives til `004 data/Analyseklart datasett/laks_ukentlig.csv` (rent) og `laks_ukentlig_features.csv` (med lag/sesong/endring). Periode: 2010-01-04 til 2026-03-09 (845 uker).

## Eksisterende kolonner i `laks_ukentlig_features.csv` (32 stk)

```
iso_aar, iso_uke, uke_start, uke_kode,
eksport_volum_tonn, eksport_pris_nok_kg,
eur_nok_snitt, eur_nok_ukeslutt, usd_nok_snitt, usd_nok_ukeslutt,
fao_global_atlantisk_tonn, fao_norge_tonn, fao_eks_norge_tonn,
pris_lag_1, pris_lag_2, pris_lag_4, pris_lag_8, pris_lag_12,
volum_lag_1, volum_lag_4, volum_lag_12,
pris_ma_4, pris_ma_12,
pris_endring_1u, pris_endring_4u, pris_endring_52u,
eur_endring_4u, usd_endring_4u,
maaned, kvartal, uke_sin, uke_cos
```

Detaljer i `004 data/Analyseklart datasett/LES_MEG.md`.

## Dine oppgaver (i prioritert rekkefølge)

1. **Differanser i volum:** `volum_endring_1u`, `volum_endring_4u`, `volum_endring_52u` – analoge til prisendringene.
2. **EUR/USD-ratio:** `eur_usd_ratio = eur_nok_snitt / usd_nok_snitt` – fanger relativ valutaeffekt.
3. **Akkumulert volum:** rullende sum av `eksport_volum_tonn` over 4, 12, 52 uker (`volum_sum_4u`, `volum_sum_12u`, `volum_sum_52u`). Bruk shift(1) først så vi unngår leakage.
4. **Glidende snitt på volum:** `volum_ma_4`, `volum_ma_12` (lag-1-basert, slik som pris_ma_*).
5. **Volatilitet:** rullende standardavvik på pris siste 4 og 12 uker (`pris_std_4`, `pris_std_12`), lag-1-basert.
6. **Forhold spot/MA:** `pris_vs_ma_12 = pris_lag_1 / pris_ma_12` – relativ avvik fra trend.

For hvert nytt feature: kommenter i koden hvilken horisontkonsistent shift som er brukt (slik at fremtidige verdier aldri lekker inn). Verifiser med en assert-test.

## Filer du eier (endre fritt)

- `004 data/Analyseklart datasett/bygg_datasett.py` – legg til feature-beregningene i samme stil som eksisterende blokk
- `004 data/Analyseklart datasett/laks_ukentlig_features.csv` – regenereres når skriptet kjøres
- `004 data/Analyseklart datasett/LES_MEG.md` – oppdater "Tilleggskolonner"-listen med de nye kolonnene

## Strenge regler

- **Aldri rename eller slett eksisterende kolonner.** Spor A og Spor B leser fra denne CSV-en parallelt. Endring av eksisterende kolonner bryter deres pipelines.
- **Aldri introduser leakage.** Alle nye features må kun bruke informasjon kjent ved tidspunkt `t-1` eller tidligere når de evalueres for tidspunkt `t`. Bruk `.shift(1)` før rullende operasjoner.
- **Behold ukentlig W-MON-frekvens og samme antall rader.** 845 uker, samme tidsindeks.
- **Ikke endre `eksport_pris_nok_kg`** (målvariabel) eller noen av lag-/endrings-kolonnene som eksisterer fra før.

## Filer du IKKE skal røre

- Alt i `006 analyse/` – tilhører Spor A og B
- Andre rådata-filer i `004 data/` enn det `bygg_datasett.py` selv leser

## Slik oppdaterer du delt status

Etter at du har generert ny features-CSV og verifisert (245 → 245+N kolonner, ingen NaN i nye kolonner utover de første ~52 ukene fra lag-effekter):

1. Oppdatér `004 data/Analyseklart datasett/LES_MEG.md` – legg de nye kolonnene under "Tilleggskolonner".
2. Oppdatér `006 analyse/LES_MEG.md` – legg en linje under "Status" om at nye features er tilgjengelig per dato, og fjern relevante punkt fra "Neste steg" ("Ekstra feature engineering").
3. Si fra til Spor A og B at de kan pulle og bruke de nye kolonnene.

Hvis du får merge-konflikt på LES_MEG.md, behold begge endringene.

## Suksesskriterium

`laks_ukentlig_features.csv` har minst 8 nye kolonner, samme antall rader som før (845), ingen leakage (alle nye features kun avhengig av `t-1` og tidligere ved tidspunkt `t`), og dokumentert i begge LES_MEG-filer.
