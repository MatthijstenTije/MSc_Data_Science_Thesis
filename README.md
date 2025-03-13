# Investigating Gender Bias in Large Language Models Through Dutch Text Generation

This repository contains the research code and documentation for the thesis project _"Investigating Gender Bias in Large Language Models Through Dutch Text Generation."_ The study is an replication of prior work on gender bias in English LLM-generated text by exploring whether similar bias patterns persist in Dutch—a language that lacks explicit grammatical gender in adjectives.

## Overview
The project investigates how gender bias emerges in large language models (LLMs) when generating Dutch text. The study replicates and extends the methodology of Soundararajan & Delany (2024) by: 
1. Constructing a Dutch lexicon of gendered adjectives using embedding-based metrics (RIPA and ML-EAT).
2. Generating sentences that describe male and female subjects using these adjectives.
3. Evaluating bias using both classifier-based methods (TPR-gap via fine-tuned BERT-based classifiers) and distribution-based metrics (Odds Ratio).

The primary research question is:
**To what extent does the gender bias detected in English LLM-generated text persist in Dutch LLM-generated text?**