# Investigating Gender Bias in Large Language Models Through Dutch Text Generation

This repository contains the research code and documentation for the thesis project _"Investigating Gender Bias in Large Language Models Through Dutch Text Generation."_ The study is a replication of prior work on gender bias in English LLM-generated text by exploring whether similar bias patterns persist in Dutch—a language that lacks explicit grammatical gender in adjectives.

## Overview
The project investigates how gender bias emerges in large language models (LLMs) when generating Dutch text. The study replicates and extends the methodology of Soundararajan & Delany (2024) by:
1. Constructing a Dutch lexicon of gendered adjectives using embedding-based metrics (RIPA and ML-EAT).
2. Generating sentences that describe male and female subjects using these adjectives.
3. Evaluating bias using both classifier-based methods (TPR-gap via fine-tuned BERT-based classifiers) and distribution-based metrics (Odds Ratio).

**Primary Research Question:**
> To what extent does the gender bias detected in English LLM-generated text persist in Dutch LLM-generated text?

---

## Phase 1

### Bijvoegelijke Naamwoorden
In deze studie is allereerst een set aan (kandidaat-)adjectieven gegenereerd uit het GiGaNT-corpus, waarbij specifiek gefilterd werd op de tag `AA(degree=pos,case=norm,infl=e)` om uitsluitend bijvoeglijke naamwoorden bekend binnen GiGaNT te selecteren.
In deze studie is allereerst een set aan (kandidaat-)adjectieven gegenereerd uit het `GiGaNT-corpus`, waarbij specifiek gefilterd werd op de tag `AA(degree=pos,case=norm,infl=e)` om uitsluitend bijvoeglijke naamwoorden bekend binnen GiGaNT te selecteren. 

__Bij gebruik van ANW adjectieven, mistten er een aantal gangbare adjectieven als: sterk, dominant, aardig, knap en schattig. Hierdoor is de keuze op de GiGaNT corpus gevallen__ 

De ruwe output is opgeschoond door:
- Enkel lemma’s met een minimale lengte van twee karakters te behouden (16253 Adjectieven)
- Gecontroleerd, of de lemma's aanwezig waren in een vooraf getraind embeddingmodel, `word2vec` Nederlands (4025 onbekende adjectieven)
- De resterende set is met behulp van het spaCy-model “nl_core_news_sm” geverifieerd op part-of-speech: enkel lemma’s die door spaCy als
  bijvoeglijk naamwoord (“ADJ”) werden geclassificeerd
- Opschoning is gedaan om woorden startend met een cijfer eruit te filteren
- Duplicaten te verwijderen.

Aansluitend is een specifieke lijst met zogenoemde *target words* (zoals “man”, “vrouw”, “jongen”, “meisje”, etc.) toegepast. Deze zijn gebruikt om woorden die inherent verbonden zijn met deze targets later uit de dataset te verwijderen.

Dit resulteerde in een voorlopige lijst van **6.439 adjectieven** die weliswaar als bijvoeglijk naamwoord kunnen functioneren, maar niet per se gangbaar of hedendaags bleken te zijn (bijvoorbeeld vanwege historische of zelden gebruikte vormen).


##### Belang van Opschoning
> In een eerdere versie van het proces, waarbij de adjectievenlijst **niet** was opgeschoond naar uitsluitend persoonlijke bijvoeglijke naamwoorden, leidde de vergelijking tot ongewenste en scheve resultaten.  
> 
> Zo kwamen bij mannelijke targets woorden als _**dood**_ en _**autodidact**_ naar voren, terwijl vrouwelijke targets gedomineerd werden door seksueel geladen of stereotyperende adjectieven zoals:
> 
> - **sensueel**: –0.173  
> - **genitaal**: –0.173  
> - **feministisch**: –0.168  
> - **erotisch**: –0.162  
> Deze bevinding onderstreept het belang van een zorgvuldige filtering op persoonlijke en hedendaagse adjectieven om betekenisvolle en eerlijke vergelijkingen te kunnen maken.


### Persoonlijke Bijvoegelijke Naamwoorden
Omdat OpenDutchWordNet gebrekkige dekking toonde voor sommige courante of hedendaagse lemma’s (zoals “zorgzaam” en “sterk”), is aanvullend het **Corpus Hedendaags Nederlands** geraadpleegd.
- Enkel adjectieven met een **minimale frequentie > 1** per lemma zijn behouden.
- Bij de zoekopdrachten is rekening gehouden met de aanwezigheid van dezelfde target words (zoals “man”, “vrouw”) om bias in context te vermijden.

In de praktijk is dit gerealiseerd via een query waarbij:
- Het bijvoeglijk naamwoord en het target word **binnen een straal van nul tokens** van elkaar voorkomen.
- Alleen combinaties met een frequentie boven een ingestelde drempel, drempel = 0 tokens, zijn behouden.
  - Deze nul kan beschouwd worden als een hyperparameter, maar enig getal boven de nul, resulteerde in een stijging van de bijvogelijke naamwoorden, die niet persoonsgeboden waren. 
- De selectie is beperkt tot `"Dutch (Dutch)"` om variatie vanuit Belgische of Surinaamse bronnen te vermijden.

Deze query resulteerde in een dataset van **13.696 combinaties van adjectieven en target words**, die opnieuw opgeschoond moeten worden.

#### Corpus Hedendaags Nederlands – Query Resultaat

```txt
Language (Common): Dutch (Dutch)
Geselecteerd subcorpus:
Totaal aantal documenten: 3.042.708 (31.2%)
Totaal aantal tokens:  1.252.988.976   (41.3%)

```txt
(
  (
    (
      [lemma="(
      __set of lemmas__
      )" & pos="aa" & pos_degree="pos|comp|sup"]
      []{0}
      (
        [lemma="(man|mannen|jongen|kerel|vader|zoon|vent)"] |
        [lemma="(vrouw|vrouwelijk|vrouwen|meisje|dame)"] 
 
      )
    )
    |
    (
      (
        [lemma="(man||jongen|kerel|vader|zoon|vent)"] |
        [lemma="(vrouw|vrouwelijk||meisje|dame)"] 
  
      )
      []{0}
      [lemma="(
        __set of lemmas__
      )" & pos="aa" & pos_degree="pos|comp|sup"]
    )
  )
)
within <s/>
```


