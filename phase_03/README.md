# Gender Stereotype Classification with Transformer Models

This repository contains three Jupyter notebooks—one per model—for fine-tuning and evaluating transformer-based classifiers that detect gender-stereotype versus counter-stereotype language in Dutch LLM outputs. Each notebook was executed on a system with two NVIDIA RTX A5000 GPUs (24 GB VRAM each).

---

## 🔍 Project Overview

We cast gender-stereotype detection as a binary classification:

- **Stereotype-congruent (1)**  
  - MM: men with masculine adjectives  
  - FF: women with feminine adjectives  
- **Stereotype-incongruent (0)**  
  - MF: men with feminine adjectives  
  - FM: women with masculine adjectives  

Our 2×2 experimental design isolates the impact of:

1. **Language coverage**: multilingual (DistilBERT) vs. monolingual Dutch (BERTje, RobBERT-2023)  
2. **Model scale**: base-scale (~110 M parameters) vs. large-scale (~350 M parameters)

All three models are fine-tuned with identical hyperparameter-search and cross-validation settings.

---

## 📚 Notebooks

Each notebook encapsulates data loading, tokenization, hyperparameter tuning (Optuna/TPE), training, and evaluation (including TPR-gap computation).

- **notebooks/Classifier_DistilBERT.ipynb**  
  Pretrained on 104 languages (DistilBERT); tests reliance on universal patterns.  
- **notebooks/Classifier_BERT.ipynb**  
  Dutch-only model (BERTje); measures gains from monolingual specialization.  
- **notebooks/Classifier_RobBERT.ipynb**  
  Dutch-only, large-scale (RobBERT-2023); evaluates the effects of model size and up-to-date data.

---

## ⚙️ Environment & Dependencies

- **Python** ≥ 3.8  
- **PyTorch** ≥ 1.10  
- **Transformers** ≥ 4.6  
- **Optuna** ≥ 3.0  
- **CUDA**-enabled drivers for NVIDIA RTX A5000  

Install with:

```bash
pip install -r requirements.txt
```


## Quick Start

Clone the repo
```bash 
git clone https://github.com/your-username/gender-stereotype-classification.git

```
Launch a Jupyter session and run the desired notebook under notebooks/:
- Classifier_DistilBERT.ipynb
- Classifier_BERT.ipynb
- Classifier_RobBERT.ipynb
Each notebook walks through:

1. Data ingestion (.csv with sentence, label, gender, template)
2. 5-fold stratified cross-validation & Optuna hyperparameter search
3. Calculation of accuracy, precision, recall, F1, and TPR-gap
4. Model checkpoints, logs, and evaluation metrics are saved in each notebook’s output directory (e.g. outputs/distilbert/).
