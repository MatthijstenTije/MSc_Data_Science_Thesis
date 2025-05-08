# 🧠 Gender Bias in Dutch LLMs: Embeddings, Pretraining, and Alignment

This repository contains the code, data, and visualizations for the thesis **"Investigating Gender Bias in Large Language Models Through Dutch Text Generation"**, submitted as part of the MSc in Data Science & Society at Tilburg University.

## 📄 Abstract

This project investigates gender bias in Dutch LLMs by analyzing three major components:

1. **Static Word Embeddings** (FastText & Word2Vec): To identify gender-coded adjectives.
2. **Sentence Generation**: Using four LLaMA-based models under varied alignment (pretrained, RLHF, QA-tuned) and temperature settings.
3. **Bias Classification**: Leveraging RobBERT, BERTje, and DistilBERT to quantify downstream bias using TPR-gap metrics.

Results show persistent masculine bias, the impact of alignment strategies, and the amplification of bias in counter-stereotypical contexts.

