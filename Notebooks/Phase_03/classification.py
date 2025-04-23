import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import optuna
from optuna.pruners import MedianPruner
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load and prepare the dataset
def load_data(file_path="Notebooks/Phase_02/output/sentences_cleaned.csv"):
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Add label column based on gender combinations
    df["label"] = df.apply(
        lambda row: "MM" if row["noun_gender"] == "male" and row["adjective_gender"] == "male"
        else "FF" if row["noun_gender"] == "female" and row["adjective_gender"] == "female"
        else "MF" if row["noun_gender"] == "male" and row["adjective_gender"] == "female"
        else "FM", axis=1
    )
    
    # Create binary stereotype column: 1 for stereotypical (MM/FF), 0 for non-stereotypical (MF/FM)
    df["stereotype"] = df["label"].apply(lambda x: 1 if x in ["MM", "FF"] else 0)
    
    # Create stratification groups for cross-validation
    df["stratify_group"] = df["stereotype"].astype(str) + "_" + df["model"].astype(str) + "_" + df["temperature"].astype(str)
    
    sentences = df["sentence"].tolist()
    labels = df["stereotype"].astype(int).tolist()
    stratify_labels = df["stratify_group"].tolist()
    
    return sentences, labels, stratify_labels, df

# Custom Dataset class
class BiasDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Training function
def train(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimization step
        optimizer.step()
        optimizer.zero_grad()
        
        # Learning rate scheduling (if provided)
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            
            preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
    
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

# Objective function for Optuna
def objective(trial, model_name, tokenizer_name, train_texts, train_labels, val_texts, val_labels):
    # Define hyperparameter search space
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    max_length = trial.suggest_categorical("max_length", [128, 256])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Initialize datasets and dataloaders
    train_dataset = BiasDataset(train_texts, train_labels, tokenizer, max_len=max_length)
    val_dataset = BiasDataset(val_texts, val_labels, tokenizer, max_len=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Calculate total steps for learning rate scheduler
    total_steps = len(train_loader) * 3  # 3 epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Linear warmup then linear decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_ratio
    )
    
    best_f1 = 0
    patience_counter = 0
    patience = 2  # Early stopping patience
    
    # Training loop
    for epoch in range(3):
        logger.info(f"Epoch {epoch + 1}")
        train_loss = train(model, train_loader, optimizer, device, scheduler)
        metrics = evaluate(model, val_loader, device)
        
        # Report to Optuna
        trial.report(metrics["f1_score"], epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        # Early stopping
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    return metrics["f1_score"]

# Main hyperparameter tuning function
def run_hyperparameter_tuning(model_name, tokenizer_name, sentences, labels, stratify_labels, n_trials=50):
    logger.info(f"Running hyperparameter tuning for {model_name}")
    
    # Create directories for saving results
    os.makedirs("tuning_results", exist_ok=True)
    os.makedirs("trained_models", exist_ok=True)
    
    # Set up cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    best_params = {}
    
    # Run hyperparameter tuning per fold
    for fold, (train_idx, test_idx) in enumerate(skf.split(sentences, stratify_labels)):
        logger.info(f"Fold {fold + 1}")
        
        # Split data
        train_texts = [sentences[i] for i in train_idx]
        test_texts = [sentences[i] for i in test_idx]
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]
        
        # Further split train data for validation during tuning
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_split = list(inner_skf.split(train_texts, [train_labels[i] for i in range(len(train_labels))]))
        inner_train_idx, val_idx = inner_split[0]
        
        inner_train_texts = [train_texts[i] for i in inner_train_idx]
        inner_train_labels = [train_labels[i] for i in inner_train_idx]
        val_texts = [train_texts[i] for i in val_idx]
        val_labels = [train_labels[i] for i in val_idx]
        
        # Create and run study
        study_name = f"{model_name.replace('/', '_')}_fold_{fold}"
        storage_name = f"sqlite:///tuning_results/{study_name}.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            load_if_exists=True
        )
        
        study.optimize(
            lambda trial: objective(
                trial, model_name, tokenizer_name, 
                inner_train_texts, inner_train_labels, 
                val_texts, val_labels
            ),
            n_trials=n_trials
        )
        
        # Save best parameters
        best_params[f"fold_{fold}"] = study.best_params
        logger.info(f"Best parameters for fold {fold + 1}: {study.best_params}")
        logger.info(f"Best F1 score: {study.best_value}")
        
        # Train final model with best parameters
        best_trial = study.best_trial
        
        # Apply best hyperparameters
        batch_size = best_trial.params["batch_size"]
        learning_rate = best_trial.params["learning_rate"]
        weight_decay = best_trial.params["weight_decay"]
        max_length = best_trial.params["max_length"]
        warmup_ratio = best_trial.params["warmup_ratio"]
        
        logger.info(f"Training final model for fold {fold + 1} with best parameters")
        
        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Create datasets and dataloaders
        train_dataset = BiasDataset(train_texts, train_labels, tokenizer, max_len=max_length)
        test_dataset = BiasDataset(test_texts, test_labels, tokenizer, max_len=max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Calculate total steps for learning rate scheduler
        total_steps = len(train_loader) * 3  # 3 epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=warmup_ratio
        )
        
        # Training final model
        for epoch in range(3):
            logger.info(f"Final training - Epoch {epoch + 1}")
            train_loss = train(model, train_loader, optimizer, device, scheduler)
        
        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, device)
        logger.info(f"Test set metrics: {test_metrics}")
        
        # Save model
        save_dir = f"trained_models/{model_name.replace('/', '_')}/fold_{fold}"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        # Add test results
        result = {
            "model": model_name,
            "fold": fold + 1,
            **test_metrics,
            "best_params": best_trial.params
        }
        fold_results.append(result)
        
    # Save all results
    results_df = pd.DataFrame(fold_results)
    results_file = f"tuning_results/{model_name.replace('/', '_')}_results.csv"
    results_df.to_csv(results_file, index=False)
    
    # Save best parameters
    with open(f"tuning_results/{model_name.replace('/', '_')}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    
    # Calculate and log average metrics
    logger.info("\nAverage performance metrics:")
    avg_metrics = {
        "accuracy": results_df["accuracy"].mean(),
        "precision": results_df["precision"].mean(),
        "recall": results_df["recall"].mean(),
        "f1_score": results_df["f1_score"].mean()
    }
    logger.info(f"Average metrics: {avg_metrics}")
    
    return results_df, best_params

# Main execution
if __name__ == "__main__":
    # Load data
    sentences, labels, stratify_labels, df = load_data()
    
    # Define models to evaluate
    models = {
        "GroNLP/bert-base-dutch-cased": "GroNLP/bert-base-dutch-cased",
        "bert-base-multilingual-cased": "bert-base-multilingual-cased",
        "DTAI-KULeuven/robbert-2023-dutch-large": "DTAI-KULeuven/robbert-2023-dutch-large"
    }
    
    all_results = []
    all_best_params = {}
    
    # Run hyperparameter tuning for each model
    for model_name, tokenizer_name in models.items():
        logger.info(f"\nRunning full pipeline for {model_name}")
        results_df, best_params = run_hyperparameter_tuning(
            model_name, tokenizer_name, sentences, labels, stratify_labels, n_trials=20
        )
        all_results.append(results_df)
        all_best_params[model_name] = best_params
    
    # Combine and save all results
    final_results = pd.concat(all_results)
    final_results.to_csv("all_models_results.csv", index=False)
    
    # Calculate and print average metrics per model
    print("\nFinal average metrics per model:")
    summary = final_results.groupby("model").agg({
        "accuracy": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1_score": ["mean", "std"]
    })
    print(summary)
    
    # Save summary
    summary.to_csv("model_comparison_summary.csv")
    
    # Save all best parameters
    with open("all_models_best_params.json", "w") as f:
        json.dump(all_best_params, f, indent=4)