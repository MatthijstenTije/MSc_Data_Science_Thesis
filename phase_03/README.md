# Phase 3: Classifier Evaluation of Gender Bias in Dutch LLM Outputs

This phase evaluates **downstream gender bias** in sentences generated by Dutch language models, using transformer-based classifiers. We quantify fairness via **true positive rate (TPR) gaps**, comparing model performance on stereotype-confirming and stereotype-violating sentences.

---

## 💡 Objective

To assess whether gender bias observed in generated sentences persists in downstream classification, and how it varies across:

- Model type (DistilBERT, BERTje, RobBERT)
- Alignment and temperature of the generating LLM
- Stereotypical vs. counter-stereotypical sentence structure

---

## 🔍 Workflow Summary

1. **Sentence Input**: Generated and validated Dutch sentences from Phase 2.
2. **Classification Models**:
   - `RobBERT-2023`
   - `BERTje`
   - `DistilBERT`
3. **Cross-Validation**:
   - 5-fold CV with Optuna hyperparameter tuning (batch size, learning rate, weight decay).
4. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1
   - TPR Gaps per gender group

---

## Files

```
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
```

---

## Metrics & Analysis

**TPR Gap Formula**:
 $$TPR Gap = TPR_male - TPR_female$$


- `TPR > 0`: Model favors male-associated sentences
- `TPR < 0`: Model favors female-associated sentences
- `TPR ≈ 0`: Model treats both fairly

**Borda Ranking**: Used to compare models across both performance and fairness dimensions.

---

## 📦 Outputs

- `.csv` summaries for accuracy and TPR gap
- `.png` or `.pdf` plots for:
  - Accuracy by temperature, model, and prompt type
  - Bias gap evolution (stereotypical vs. counter-stereotypical)
- Notebook logs with Optuna optimization and final metrics

---