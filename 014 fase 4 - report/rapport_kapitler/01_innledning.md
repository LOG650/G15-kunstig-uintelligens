# 1. Innledning

## 1.1 Bakgrunn og motivasjon

Norsk lakseoppdrett er en av landets største eksportnæringer, med en eksportverdi som i 2023 oversteg 100 milliarder kroner (SSB, 2024). Eksportprisen for fersk laks svinger kraftig fra uke til uke og er avgjørende for lønnsomheten hos både oppdrettere, eksportører og kjøpere. Aktører med eksponering mot spotmarkedet har behov for pålitelige prognoser på 4–12 ukers sikt for å planlegge slakting, logistikk og prissikring.

Til tross for den kommersielle viktigheten er offentlig tilgjengelig forskning på kortsiktig ukentlig lakseprisprognosering begrenset. FAO publiserer kvartalsvise prisindekser, og SSB rapporterer ukentlig; men koblingen mellom disse kildene og maskinlæringsbaserte prognosemetoder er lite utforsket i litteraturen.

## 1.2 Problemstilling

Rapporten besvarer følgende spørsmål:

> **Hvilke modeller gir lavest prediksjonsfeil (MAE) for ukentlig eksportpris på fersk norsk laks over prognosehorisonter på 4, 8 og 12 uker?**

Som underspørsmål undersøkes:

1. Er statistiske tidsseriemodeller (SARIMA/SARIMAX) bedre enn gradientøkende tremodeller på korte horisonter?
2. Gir kombinasjon av EUR/USD-valutakurs som eksogen variabel (SARIMAX) bedre prediksjoner enn SARIMA alene?
3. Hvor godt kalibrerte er modellenes konfidensintervaller?
4. Hvilke features er viktigst i de maskinlæringsbaserte modellene?

## 1.3 Avgrensning

Studien dekker ukentlige data fra 2009 til 2024 og evaluerer modellene på de siste 104 ukene (~2 år) i en walk-forward-oppsett uten fremtidig informasjonslekasje. Det er ikke gjort markedsanalyse eller optimert handlingsstrategi — fokus er utelukkende på statistisk prognose.

Rapporten inngår som avsluttende prosjektarbeid i emnet LOG650 (Logistikk og kunstig intelligens) ved Høgskolen i Molde (HiM).
