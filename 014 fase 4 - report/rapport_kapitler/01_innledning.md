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

## 1.4 Relatert forskning

### Lakseprisens dynamikk og markedsstruktur

Norsk lakseoppdrett og prisdannelsen i dette markedet er grundig studert av Asche og medarbeidere. Asche og Bjørndal (2011) dokumenterer strukturen i atlantisk laksemarked, herunder prisintegrasjon mellom europeiske markeder og rollen til valutakurser som prisstimulator. Oglend (2013) viser at lakseprisvolatiliteten har økt over tid og identifiserer ikke-linearitet i prisdynamikken — funn som motiverer bruken av ikke-lineære maskinlæringsmodeller i tillegg til klassiske statistiske metoder. Dahl og Oglend (2014) finner at lakseprisen er integrert med EUR/USD-kursen, noe som er den empiriske begrunnelsen for å inkludere valutakurs som eksogen variabel i SARIMAX-spesifikasjonen i denne studien.

### SARIMA og statistiske tidsseriemodeller for råvarer

Box, Jenkins, Reinsel og Ljung (2015) er standardverket for SARIMA-modellering og etablerer rammeverket denne studien bygger på. Hyndman og Athanasopoulos (2021) gir en oppdatert gjennomgang av prognosemetoder og konkluderer at sesongmodeller med differensiering (SARIMA-familien) generelt presterer godt for råvarepriser med stabile sesongmønstre, men at de er sårbare for strukturelle brudd. Begge disse referansene støtter valg av SARIMA som statistisk referansemodell i studien.

### Gradientøkende tremodeller og ensemble-metoder for tidsserier

Chen og Guestrin (2016) introduserte XGBoost, og Ke et al. (2017) introduserte LightGBM, begge med dokumentert overlegen ytelse på tabelldata sammenlignet med dypere nevrale nett ved moderate datasettsstørrelser. For tidsserieprognose spesifikt fant Makridakis, Spiliotis og Assimakopoulos (2020) i den store M4-konkurransen (100 000 tidsserier, 61 metoder) at kombinasjonsmodeller konsekvent overgår enkeltmodeller, og at hybridmodeller som kombinerer statistiske og maskinlæringsbaserte metoder hevder seg blant de beste. Dette er en direkte motivasjon for ensemble-tilnærmingen i denne studien.

### Forskningsgap

Til tross for den kommersielle viktigheten av lakseprognosering er litteraturen på *ukentlig* lakseprognosering med maskinlæringsbaserte metoder begrenset. Eksisterende studier fokuserer primært på månedlig eller kvartalsvis prisdynamikk og markedsintegrasjon (Asche et al.), eller på bredere råvaremarkeder (Makridakis et al.). Kombinasjonen av SSBs ukentlige eksportdata, FAO-prisindeks og gradientøkende ensembler i en walk-forward-evaluering for norsk laks er ikke tidligere dokumentert i den åpent tilgjengelige litteraturen, og utgjør dette studiet sitt empiriske bidrag.
