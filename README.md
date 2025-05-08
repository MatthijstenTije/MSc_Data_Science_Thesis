# Dutch Adjective Gender Bias Detection

A Thesis Project for the MSc in Data Science & Society – Tilburg University

This repository supports the thesis titled:
__“Investigating Gender Bias in Large Language Models Through Dutch Text Generation: Assessing the Role of Embeddings, Pretraining, and Alignment”__,  submitted in partial fulfillment of the requirements for the Master of Science in Data Science & Society at Tilburg University, School of Humanities and Digital Sciences.

---

## Table of Contents

- [Dutch Adjective Gender Bias Detection](#dutch-adjective-gender-bias-detection)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation and Requirements](#installation-and-requirements)
  - [📥 How to Download and Run LLaMA Models (Local)](#-how-to-download-and-run-llama-models-local)
    - [Step-by-Step Instructions:](#step-by-step-instructions)
  - [Data Sources](#data-sources)
  - [Analysis Workflow](#analysis-workflow)
  - [🧾 Source, Ethics, Code, and Technology Statement](#-source-ethics-code-and-technology-statement)
    - [📚 Data Source](#-data-source)
    - [⚖️ Ethics Statement](#️-ethics-statement)

---

## Overview

This research investigates **gender bias in Dutch Large Language Models (LLMs)** by analyzing how gendered connotations are embedded in and reproduced by language models.

Building on English-language findings, this study asks whether similar biases arise in Dutch—a language without grammatical gender marking in adjectives.

The analysis unfolds in three main phases:

1. **Embedding Analysis**  
   - Identify gender-coded Dutch adjectives using FastText and Word2Vec embeddings  
   - Apply cosine similarity and RIPA metrics to detect and quantify bias  

2. **Controlled Sentence Generation**  
   - Generate 16,000 Dutch sentences using four LLaMA-based models (via Ollama)  
   - Vary alignment (e.g., RLHF, QA tuning) and sampling temperature  

3. **Classifier Evaluation**  
   - Use RobBERT, BERTje, and DistilBERT to assess stereotype consistency  
   - Quantify downstream bias via True Positive Rate (TPR) gaps  

Bias patterns are statistically tested and visualized, demonstrating that gender associations in embeddings persist and propagate through generation and classification pipelines.

---

## Project Structure

```
├── data
├── phase_01
│   ├── output
│   ├── bias_metrics.py
│   ├── config.py
│   ├── main.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── README.md
│   ├── ripa_analysis.py
│   ├── utils.py
│   └── visualization.py
├── phase_02
│   ├── logs
│   │   ├── llama2-uncensored
│   │   ├── llama3_8b
│   │   └── llama3_text
│   ├── llama3-chatqa_8b
│   │   ├── main.log
│   │   └── quality_issues.json
│   ├── output
│   │   └── intermediate
│   │       └── llama3-chatqa_8b_female_female_temp1_5_state_1745780595.json
│   ├── prompts
│   │   ├── prompt_femaleNouns_femaleAdjs.txt
│   │   ├── prompt_femaleNouns_maleAdjs.txt
│   │   ├── prompt_maleNouns_femaleAdjs.txt
│   │   └── prompt_maleNouns_maleAdjs.txt
│   ├── visualizations
│   ├── 01_async_generate_sentences.py
│   ├── 02_convert_logs_to_csv.py
│   ├── 03_clean_sentences.py
│   ├── 04_gender_summary.py
│   ├── 05_temp_gender_summary.py
│   ├── 06_model_gender_summary.py
│   ├── 07_word_counts.py
│   ├── 08_iterations_visualization.py
│   ├── 09_error_analysis.py
│   ├── 10_leakage.py
│   ├── config.py
│   ├── old_generate_sentences.py
│   └── README.md
├── phase_03
│   ├── visualizations
│   │   ├── Classifier_BERT.ipynb
│   │   ├── Classifier_DistillBERT.ipynb
│   │   ├── Classifier_RobBERT.ipynb
│   │   ├── fold_visualization.py
│   │   ├── make_accuracy_summary.py
│   │   ├── make_borda_ranking.py
│   │   ├── make_trp_gap_summary.py
│   │   ├── visualization_accuracy.py
│   │   └── visualization_tpr_gap.py
│   ├── README.md
│   └── requirements.txt
├── .gitignore
├── README.md
└── requirements.txt

```

---

## Installation and Requirements

**Python Version**: 3.7 or later is recommended.

Install all required dependencies:

```bash
pip install -r requirements.txt
python -m spacy download nl_core_news_lg
```

---

## 📥 How to Download and Run LLaMA Models (Local)

For reproducible generation of Dutch text, this project uses four variants of Meta’s LLaMA models, executed locally via [Ollama](https://ollama.com):

### Step-by-Step Instructions:

1. **Install Ollama** (macOS, Linux, or Windows):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Run LLaMA model variants**:
   ```bash
   ollama run llama3
   ollama run llama3:chat
   ollama run llama3:text
   ollama run llama2-uncensored
   ```

3. **Configuration**:
   - Use fixed seeds and temperatures in prompt scripts for consistency.
   - Prompts are in `phase_02/prompts/`.

⚠️ **Note**: These models may generate biased content. Dataset is not shared publicly but available on academic request.

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

## 🧾 Source, Ethics, Code, and Technology Statement

### 📚 Data Source

- **GiGaNT corpus** from [Institute for the Dutch Language](https://ivdnt.org/corpora-lexica/gigant/)
- **CHN corpus** via [CLARIN portal](https://portal.clarin.ivdnt.org/corpus-frontend-chn/chn-extern/search/)
- **Embeddings**:
  - [FastText cc.nl.300.vec.gz](https://fasttext.cc/docs/en/crawl-vectors.html)
  - [Word2Vec (SoNaR)](https://github.com/clips/dutchembeddings)
- **Sentence generation**:
  - 16,000 sentences generated using LLaMA models locally via Ollama
- **Transformers for bias detection**:
  - [DistilBERT](https://huggingface.co/distilbert-base-multilingual-cased)
  - [BERTje](https://huggingface.co/GroNLP/bert-base-dutch-cased)
  - [RobBERT 2023](https://huggingface.co/DTAI-KULeuven/robbert-2023-dutch-large)
  - Executed on Runpod.io using NVIDIA A5000 GPUs

All figures and outputs were generated by the author.

---

### ⚖️ Ethics Statement

This project includes generated sentences reflecting gender stereotypes for research purposes. While this helps identify and quantify bias, it may also unintentionally replicate harmful representations.

To mitigate risk:
- Dataset is **restricted** and only available for academic purposes upon request.
- Generation was **controlled and reproducible**.
- Use of these materials for reinforcement or deployment without context is **explicitly discouraged**.

---
