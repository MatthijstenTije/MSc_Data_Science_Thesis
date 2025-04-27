import os
import logging
from datetime import datetime
import random                     # NEW
import pandas as pd
import numpy as np
import torch
import optuna                    # NEW
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit  # CHANGED
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

SEED = 42                      
random.seed(SEED)              
np.random.seed(SEED)             
torch.manual_seed(SEED)           

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load and prepare data ---------------------------------------------------
# ---------------------------------------------------------------------------
def load_data(path="Notebooks/Phase_02/output/sentences_cleaned.csv"):
    df = pd.read_csv(path)
    df["label"] = df.apply(
        lambda r: "MM" if r.noun_gender == "male" and r.adjective_gender == "male"
        else "FF" if r.noun_gender == "female" and r.adjective_gender == "female"
        else "MF" if r.noun_gender == "male" and r.adjective_gender == "female"
        else "FM",
        axis=1,
    )
    df["stereotype"] = df["label"].isin(["MM", "FF"]).astype(int)
    df["stratify_group"] = (
        df["stereotype"].astype(str)
        + "_"
        + df["model"].astype(str)
        + "_"
        + df["temperature"].astype(str)
    )
    return df

# ---------------------------------------------------------------------------
# Compute metrics ---------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0 
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------------------------------------------------------------------------
# Training and evaluation -------------------------------------------------
# ---------------------------------------------------------------------------

def run_cv_hp_search(model_name, tokenizer_name, df, n_splits=5, n_trials=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(df, df["stratify_group"]), start=1
    ):
        logger.info(f"Fold {fold}/{n_splits} for {model_name}")
        train_df_full = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # --- Stratified inner split for validation -------------------------
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_indices, val_indices = next(
            sss.split(train_df_full, train_df_full["stratify_group"])
        )
        train_df = train_df_full.iloc[train_indices]
        val_df = train_df_full.iloc[val_indices]

        # Load tokenizer -----------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize(df_):
            return tokenizer(
                df_["sentence"].tolist(),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

        train_enc = tokenize(train_df)
        val_enc = tokenize(val_df)
        test_enc = tokenize(test_df)

        # Simple PyTorch Dataset -------------------------------------------
        class TorchDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels.tolist()

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        train_ds = TorchDataset(train_enc, train_df["stereotype"])
        val_ds = TorchDataset(val_enc, val_df["stereotype"])
        test_ds = TorchDataset(test_enc, test_df["stereotype"])

        # Model init --------------------------------------------------------
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )

        # Base training arguments -----------------------------------------
        base_out = f"./results/{model_name.replace('/', '_')}/fold_{fold}"
        base_log = f"./logs/{model_name.replace('/', '_')}/fold_{fold}"
        base_args = dict(
            output_dir=base_out,
            evaluation_strategy="epoch",
            save_strategy="no",         
            logging_strategy="epoch",
            logging_dir=base_log,
            report_to="none",
            num_train_epochs=3,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            dataloader_num_workers=4,
            per_device_eval_batch_size=32,
            seed=SEED,
        )

        args = TrainingArguments(per_device_train_batch_size=16, **base_args)

        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )

        # Hyperparameter search -------------------------------------------
        best_run = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=lambda trial: {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                "per_device_train_batch_size": trial.suggest_categorical(
                    "per_device_train_batch_size", [8, 16, 32]
                ),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            },
            n_trials=n_trials,
            optuna_sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        logger.info(
            f"Best hyperparameters fold {fold}: {best_run.hyperparameters}"
        )

        # Re‑instantiate trainer with best HP ------------------------------
        best_args = TrainingArguments(
            **base_args,
            per_device_train_batch_size=best_run.hyperparameters["per_device_train_batch_size"],
            learning_rate=best_run.hyperparameters["learning_rate"],
            weight_decay=best_run.hyperparameters.get("weight_decay", 0.0),
        )

        trainer = Trainer(
            model_init=model_init,
            args=best_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )

        trainer.train()

        # Evaluate on held‑out test set ------------------------------------
        metrics = trainer.evaluate(eval_dataset=test_ds)
        logger.info(
            (
                f"Test metrics fold {fold} – "
                f"Acc: {metrics['eval_accuracy']:.4f} | "
                f"Prec: {metrics['eval_precision']:.4f} | "
                f"Rec: {metrics['eval_recall']:.4f} | "
                f"F1: {metrics['eval_f1']:.4f}"
            )
        )
        results.append(metrics)

    # ---- Save once per model -------------------------------------------
    pd.DataFrame(results).to_csv(
        f"results_{model_name.replace('/', '_')}.csv", index=False
    )

    return results

# ---------------------------------------------------------------------------
# Main -------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    models = [
        ("GroNLP/bert-base-dutch-cased", "GroNLP/bert-base-dutch-cased"),
        ("bert-base-multilingual-cased", "bert-base-multilingual-cased"),
        ("DTAI-KULeuven/robbert-2023-dutch-large", "DTAI-KULeuven/robbert-2023-dutch-large"),
    ]

    summary = {}
    for model_name, tokenizer_name in models:
        logger.info(f"Starting model {model_name}")
        res = run_cv_hp_search(model_name, tokenizer_name, df, n_splits=5, n_trials=5)
        summary[model_name] = res

    print(summary)