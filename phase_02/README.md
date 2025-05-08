# Phase 2: Gender Bias in Generated Sentences

This repository contains code for **Phase 2** of the thesis project on gender bias in Dutch LLMs. It focuses on the **generation and analysis of AI-generated Dutch sentences**, exploring how alignment strategies, decoding temperatures, and prompt structures influence gender bias in outputs.

This phase builds on embedding-level bias insights (Phase 1) and precedes the classifier evaluation (Phase 3).
---

## Table of Contents

- [Phase 2: Gender Bias in Generated Sentences](#phase-2-gender-bias-in-generated-sentences)
  - [This phase builds on embedding-level bias insights (Phase 1) and precedes the classifier evaluation (Phase 3).](#this-phase-builds-on-embedding-level-bias-insights-phase-1-and-precedes-the-classifier-evaluation-phase-3)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Data Sources](#data-sources)
  - [Analysis Workflow](#analysis-workflow)
  - [Output](#output)

---

## Overview

This phase includes:

1. **Sentence Generation**:
   - Use asynchronous and batch methods to generate 16,000+ sentences using LLaMA-based language models hosted locally.

2. **Data Processing**:
   - Convert raw generation logs into structured CSVs.
   - Preprocess and clean outputs for duplicates, malformed sentences, and validation errors.

3. **Gender Mention Analysis**:
   - Track frequency of male/female-coded adjectives in generated sentences.
   - Compute co-occurrence and leakage statistics.

4. **Visualization**:
   - Plot gender proportions and word distributions over temperature, model type, and prompt condition.

5. **Leakage & Validation**:
   - Inspect failures such as adjective leakage, repeated content, malformed prompts, and semantic drift.

---


## Project Structure

```
.
├── output/
│   └── (place for generated sentences and processed CSVs)
├── prompts/
│   └── (place for prompt templates)
├── 01_async_generate_sentences.py
├── 02_convert_logs_to_csv.py
├── 03_clean_sentences.py
├── 04_gender_summary.py
├── 05_temp_gender_summary.py
├── 06_model_gender_summary.py
├── 07_word_counts.py
├── 08_iterations_visualization.py 
├── 09_error_analysis.py 
├── 10_leakage.py
├── config.py
└── README.md 
```

- **data/**: Folder containing raw logs, intermediate CSVs, and cleaned data.
- **01_async_generate_sentences.py**: Asynchronous generation of sentences using language models.
- **01b_generate_sentences.py**: Alternate version for sentence generation (non-async).
- **02_convert_logs_to_csv.py**: Converts generation logs into structured CSV files.
- **03_clean_sentences.py**: Cleans and standardizes generated sentences.
- **04_gender_summary.py**: Summarizes gender-specific mentions.
- **05_temp_gender_summary.py**: Temporary analysis of gender proportions.
- **06_model_gender_summary.py**: model-based gender bias summaries.
- **07_word_counts.py**: Analyzes word frequencies in generated texts.
- **08_iterations_visualization.py**: Visualizes trends in bias over multiple iterations.
- **09_error_analysis.py**: Error inspection and correction
- **10_leakage.py**: Leakage analysis and diagnostic tools



## Data Sources

- **Generated Sentences**: 16,000 sentences across 80 configurations (4 models × 5 temperatures × 4 prompt structures)
- **Gender Lexicons**: Adjective sets with male- or female-coded bias from Phase 1
- **Logs and Intermediate Files**: Stored in output/ during generation and cleaning
---

## Analysis Workflow

1. Generate sentences asynchronously or in batches.
2. Convert raw logs into structured CSV files.
3. Clean and filter sentences for quality and relevance.
4. Identify gender mentions using word lists.
5. Summarize gender proportions by prompt, iteration, or model.
6. Visualize bias evolution across generation rounds.
7. Investigate misclassifications and gender term mismatches.
8. Detect prompt leakage and analyze its implications

---

## Output

The analysis produces:

- **Clean CSVs** of generated and cleaned sentences.
- **Gender summary tables** showing bias proportions.
- **Word count analyses** across datasets.
- **Plots and figures** displaying bias trends over iterations.
- **Error reports** and potential corrections.
- **Leakage assessments** highlighting problematic prompts or outputs.

---

