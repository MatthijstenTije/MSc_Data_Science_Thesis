# Dutch Adjective Gender Bias Detection

This project analyzes gender bias in Dutch adjectives using FastText and Word2Vec models, leveraging both manual cosine similarity methods and the [WEFE (Word Embeddings Fairness Evaluation)](https://github.com/dccuchile/wefe) framework, including the RIPA metric.

It includes both statistical analysis and visualization of bias patterns across embeddings.

---

## Table of Contents

- [Dutch Adjective Gender Bias Detection](#dutch-adjective-gender-bias-detection)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation and Requirements](#installation-and-requirements)
  - [Data Sources](#data-sources)
  - [Analysis Workflow](#analysis-workflow)
  - [Output](#output)

---

## Overview

The code in this repository demonstrates:

1. **Loading Pre-trained Dutch Embeddings**:
   - We load FastText embeddings (`cc.nl.300.vec.gz`) and Word2Vec Sonar embeddings.

2. **Extracting and Filtering Adjectives**:
   - Parse a CSV file of Dutch phrases using SpaCy’s `nl_core_news_lg` model to **extract adjectives**.
   - Remove duplicates, filter out-of-vocabulary words, and optionally exclude words containing substrings like "man" or "vrouw".

3. **Computing Bias**:
   - Measure bias via **cosine similarity**–based metrics between adjectives and male/female target word sets.
   - Compute **p-values** via permutation tests for significance.
   - Apply **RIPA** (Relative Index of Polarization Attribute) from WEFE for a more sophisticated bias analysis.

4. **Visualizing Bias**:
   - Scatter plots comparing bias across models, highlighting significance categories.
   - Bar plots showing the top male-biased and female-biased adjectives.
   - RIPA plots to visualize residualized bias distributions.

---

## Project Structure

```
.
├── data/
│   ├── cc.nl.300.vec.gz
│   └── Corpus_Hedendaags_Nederlands_Adjectives.csv
├── main.py
├── config.py
├── models.py
├── preprocessing.py
├── bias_metrics.py
├── utils.py
├── ripa_analysis.py
├── visualization.py
├── requirements.txt
├── README.md
└── notebooks/
    └── 04_GiGant_CHN_Fasttext.ipynb
```

- **data/**: Contains Dutch FastText embeddings and the corpus CSV.
- **notebooks/**: Jupyter notebooks for exploratory or secondary analysis.
- **main.py**: Main script to run the complete bias analysis.
- **config.py**: Configuration settings and paths.
- **models.py**: Loading word embedding models.
- **preprocessing.py**: Extracting and filtering adjectives.
- **bias_metrics.py**: Computing bias metrics and permutation tests.
- **utils.py**: Helper functions (e.g., cosine similarity).
- **ripa_analysis.py**: Running RIPA-based bias analysis.
- **visualization.py**: Plotting functions for results.

---

## Installation and Requirements

**Python Version**: 3.7 or later is recommended.

Install all required dependencies:

```bash
pip install -r requirements.txt
python -m spacy download nl_core_news_lg
```

**Key Libraries**:
- [gensim](https://pypi.org/project/gensim/)
- [spacy](https://spacy.io/)
- [numpy](https://pypi.org/project/numpy/)
- [pandas](https://pypi.org/project/pandas/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/) (optional, for nicer plots)
- [wefe](https://github.com/dccuchile/wefe)

---

## Data Sources

- **Dutch FastText Embeddings**: Download from [FastText Crawl Vectors](https://fasttext.cc/docs/en/crawl-vectors.html).
- **Word2Vec Sonar Embeddings**: (Provide download link if available).
- **Dutch Corpus CSV**: `Corpus_Hedendaags_Nederlands_Adjectives.csv`, parsed to extract adjectives.

Make sure to update file paths in your scripts and notebooks to match your local environment.

---

## Analysis Workflow

1. Load the Dutch embeddings (FastText and Sonar Word2Vec).
2. Parse the Dutch corpus and extract adjectives.
3. Filter adjectives and ensure they exist in the embedding vocabulary.
4. Compute gender bias scores:
   - Cosine similarity between adjectives and male/female word sets.
   - Statistical significance via permutation tests.
   - RIPA bias analysis using WEFE.
5. Visualize results with scatter plots, bar charts, and bias distributions.

Run the full analysis:

```bash
python main.py
```

---

## Output

The analysis generates:

- **Scatter Plots**: Comparing cosine similarity–based bias scores between models.
- **Bar Plots**: Highlighting the most male-biased and female-biased adjectives.
- **RIPA Visualizations**: Showing residualized bias scores and distributions.

---
