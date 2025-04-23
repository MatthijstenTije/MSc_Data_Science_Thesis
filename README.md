# README: Dutch Adjective Bias Detection

This repository contains code for identifying and measuring **gender bias** in Dutch adjectives using **FastText** word embeddings and the [WEFE (Word Embeddings Fairness Evaluation)](https://github.com/dccuchile/wefe) framework. The analysis focuses on comparing each adjective’s similarity to two sets of target words: those related to “male” and “female” gender. Key metrics used include:

- **Cosine similarity**–based bias scores (manual method).  
- **RIPA** (Relative Index of Polarization Attribute) from WEFE.

---

## Table of Contents

- [README: Dutch Adjective Bias Detection](#readme-dutch-adjective-bias-detection)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation and Requirements](#installation-and-requirements)
  - [Data Sources](#data-sources)

---

## Overview

The code in this repository demonstrates:

1. **Loading Pre-trained Dutch FastText Embeddings**:  
   We load the embeddings from a `.vec.gz` file containing 300-dimensional vectors for Dutch words (e.g. `cc.nl.300.vec.gz`).

2. **Extracting and Filtering Adjectives**:
   - We parse a CSV file of Dutch phrases using SpaCy’s `nl_core_news_lg` model to **extract adjectives**.
   - We filter out duplicates, remove any that do not appear in the embedding vocabulary, and optionally exclude words containing certain substrings (like “man” or “vrouw”) if needed.

3. **Computing Bias**:
   - We measure how strongly each adjective is associated with male vs. female target words using **cosine similarity**.  
   - We also compute a statistic (`p_value`) using permutation tests, indicating how significant that difference is.  
   - Separately, we use **RIPA** from WEFE to capture an alternative measure of the embeddings’ bias.

4. **Visualizing Bias**:
   - We generate bar charts highlighting the **top** most biased adjectives for each gender.  
   - We provide ways to compare or investigate words of interest (e.g., “sterk,” “zacht,” “dominant,” etc.).

---

## Project Structure

A typical structure might look like:

```
.
├── data/
│   ├── cc.nl.300.vec.gz
│   └── Corpus_Hedendaags_Nederlands_Adjectives.csv
├── notebooks/
│   └── 04_GiGant_CHN_Fasttext.ipynb
├── requirements.txt
├── README.md
└── ...
```


- **data/**: Folder containing the Dutch embeddings (`cc.nl.300.vec.gz`) and a CSV of Dutch phrases to extract adjectives.  
- **notebooks/**: Contains the main Jupyter notebook (`04_GiGant_CHN_Fasttext.ipynb`).  
- **requirements.txt**: Python dependencies (see next section).  
- **README.md**: Documentation (this file).

---

## Installation and Requirements

1. **Python Version**: 3.7 or later is recommended.  
2. **Dependencies**:
   - [gensim](https://pypi.org/project/gensim/) for loading and handling word embeddings  
   - [spacy](https://spacy.io/) for text processing  
     - Also install the Dutch model: `python -m spacy download nl_core_news_lg`  
   - [numpy](https://pypi.org/project/numpy/)  
   - [pandas](https://pypi.org/project/pandas/)  
   - [matplotlib](https://matplotlib.org/)  
   - [seaborn](https://seaborn.pydata.org/) (optional, used in some plots)  
   - [wefe](https://github.com/dccuchile/wefe) for the RIPA metric

Install everything in one go:

```bash
pip install -r requirements.txt
python -m spacy download nl_core_news_lg
```

---

## Data Sources

- Dutch FastText Embeddings: FastText `cc.nl.300` (compressed .vec.gz).
- `Corpus_Hedendaags_Nederlands_Adjectives.csv`: A CSV file of Dutch phrases that will be parsed by SpaCy to extract adjectives.
- 
Make sure to update file paths in the notebook to match your local environment.