# Gender Bias Detection in Generated Sentences

This repository contains code for analyzing **gender bias** in AI-generated Dutch sentences.  
The workflow includes generating sentences, cleaning and preprocessing outputs, summarizing gender mentions, and visualizing the evolution of bias across iterations.

---

## Table of Contents

- [Gender Bias Detection in Generated Sentences](#gender-bias-detection-in-generated-sentences)
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

1. **Sentence Generation**:
   - Use asynchronous and batch methods to generate large numbers of sentences based on prompts.

2. **Data Processing**:
   - Convert raw log files into clean CSV datasets.
   - Preprocess and clean generated sentences, handling duplicates and irrelevant entries.

3. **Gender Analysis**:
   - Identify mentions of gender-specific terms in the sentences (e.g., "man", "vrouw").
   - Summarize gender bias across different prompts, models, and iterations.

4. **Visualization**:
   - Plot gender proportions across multiple rounds of generation.
   - Explore word frequency counts and trends over time.

---

## Project Structure

```
.
├── output/
│   └── (place for generated sentences and processed CSVs)
├── prompts/
│   └── (place for prompt templates)
├── 01_async_generate_sentences.py
├── 01b_generate_sentences.py
├── 02_convert_logs_to_csv.py
├── 03_clean_sentences.py
├── 04_gender_summary.py
├── 05_temp_gender_summary.py
├── 06_model_gender_summary.py
├── 07_word_counts.py
├── 08_iterations_visualization.py
├── leakage.ipynb
├── requirements.txt
└── README.md
```

- **data/**: Folder containing raw logs, intermediate CSVs, and cleaned data.
- **01_async_generate_sentences.py**: Asynchronous generation of sentences using language models.
- **01b_generate_sentences.py**: Alternate version for sentence generation (non-async).
- **02_convert_logs_to_csv.py**: Converts generation logs into structured CSV files.
- **03_clean_sentences.py**: Cleans and standardizes generated sentences.
- **04_gender_summary.py**: Summarizes gender-specific mentions.
- **05_temp_gender_summary.py**: Temporary analysis of gender proportions.
- **06_model_gender_summary.py**: Advanced model-based gender bias summaries.
- **07_word_counts.py**: Analyzes word frequencies in generated texts.
- **08_iterations_visualization.py**: Visualizes trends in bias over multiple iterations.
- **leakage.ipynb**: Notebook analyzing possible data leakage issues.

---

## Installation and Requirements

**Python Version**: 3.8 or later is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Key Libraries**:
- [asyncio](https://docs.python.org/3/library/asyncio.html) (for async sentence generation)
- [openai](https://pypi.org/project/openai/) (if generating using OpenAI models)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

---

## Data Sources

- **Generated Sentences**: Output from language models based on custom prompts.
- **Gender Word Lists**: Predefined lists of male and female-associated terms.
- **Logs and CSVs**: Intermediate outputs stored during the workflow.

---

## Analysis Workflow

1. Generate sentences asynchronously or in batches.
2. Convert raw logs into structured CSV files.
3. Clean and filter sentences for quality and relevance.
4. Identify gender mentions using word lists.
5. Summarize gender proportions by prompt, iteration, or model.
6. Visualize bias evolution across generation rounds.

---

## Output

The analysis produces:

- **Clean CSVs** of generated and cleaned sentences.
- **Gender summary tables** showing bias proportions.
- **Word count analyses** across datasets.
- **Plots and figures** displaying bias trends over iterations.

---

