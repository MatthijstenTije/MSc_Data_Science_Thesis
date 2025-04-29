import os
import logging
from datetime import datetime
import random                   
import pandas as pd
import numpy as np
import torch
import optuna                   
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit  
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import os
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
def load_data(path="phase_02/output/sentences_final.csv"):
    df = pd.read_csv(path)
    keep_columns = [
        "sentence",
        "model",
        "noun_gender",
        "adjective_gender",
        "temperature",
    ]
    df = df.loc[:, [col for col in keep_columns if col in df.columns]]
    df["label"] = df.apply(
        lambda r: "MM" if r.noun_gender == "male" and r.adjective_gender == "male"
        else "FF" if r.noun_gender == "female" and r.adjective_gender == "female"
        else "MF" if r.noun_gender == "male" and r.adjective_gender == "female"
        else "FM",
        axis=1,
    )
    # Stereotype: 1 = consistent with gender stereotype (MM, FF), 0 = contradictory (MF, FM)
    df["stereotype"] = df["label"].isin(["MM", "FF"]).astype(int)
    df["stratify_group"] = (
        df["stereotype"].astype(str)
        + "_"
        + df["model"].astype(str)
        + "_"
        + df["temperature"].astype(str)
    )
    return df

def create_model_init(model_name, num_labels=2):
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    return model_init

# ---------------------------------------------------------------------------
# Compute metrics including TPR Gap ---------------------------------------
# ---------------------------------------------------------------------------

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0 
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def compute_detailed_metrics(pred, metadata):
    """
    Berekent gedetailleerde metrics inclusief TPR Gap per gender
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    # Basis metrics
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0 
    )
    
    results = {
        "accuracy": acc, 
        "precision": prec, 
        "recall": rec, 
        "f1": f1
    }
    
    # Gender-specifieke metrics berekenen als we gender info hebben
    if 'noun_gender' in metadata.columns:
        # Maak subsets per gender en class
        male_indices = metadata['noun_gender'] == 'male'
        female_indices = metadata['noun_gender'] == 'female'
        
        # Stereotype klasse (positieve klasse = 1)
        stereotype_indices = metadata['stereotype'] == 1
        contra_indices = metadata['stereotype'] == 0
        
        # TPR voor mannelijke stereotypes (male + stereotype = 1)
        male_stereotype = male_indices & stereotype_indices
        if sum(male_stereotype) > 0:
            tpr_male_s = accuracy_score(
                labels[male_stereotype], 
                preds[male_stereotype]
            )
        else:
            tpr_male_s = 0
            
        # TPR voor vrouwelijke stereotypes (female + stereotype = 1)
        female_stereotype = female_indices & stereotype_indices
        if sum(female_stereotype) > 0:
            tpr_female_s = accuracy_score(
                labels[female_stereotype], 
                preds[female_stereotype]
            )
        else:
            tpr_female_s = 0
            
        # TPR Gap voor stereotype klasse (S)
        tpr_gap_s = tpr_male_s - tpr_female_s
        
        # TPR voor mannelijke contra-stereotypes (male + contra-stereotype)
        male_contra = male_indices & contra_indices
        if sum(male_contra) > 0:
            tpr_male_contra = accuracy_score(
                labels[male_contra], 
                preds[male_contra]
            )
        else:
            tpr_male_contra = 0
            
        # TPR voor vrouwelijke contra-stereotypes (female + contra-stereotype)
        female_contra = female_indices & contra_indices
        if sum(female_contra) > 0:
            tpr_female_contra = accuracy_score(
                labels[female_contra], 
                preds[female_contra]
            )
        else:
            tpr_female_contra = 0
            
        # TPR Gap voor contra-stereotype klasse (contra-S)
        tpr_gap_contra = tpr_male_contra - tpr_female_contra
        
        # Voeg gender-specifieke metrics toe
        results.update({
            "tpr_male_s": tpr_male_s,
            "tpr_female_s": tpr_female_s,
            "tpr_gap_s": tpr_gap_s,
            "tpr_male_contra": tpr_male_contra,
            "tpr_female_contra": tpr_female_contra,
            "tpr_gap_contra": tpr_gap_contra
        })
    
    return results

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

# ---------------------------------------------------------------------------
# Training and evaluation with TPR Gap Analysis ---------------------------
# ---------------------------------------------------------------------------

def run_gender_bias_cv(model_name, tokenizer_name, df, n_splits=5, n_trials=10):
    """
    Cross-validatie methode met hyperparameter optimalisatie op eerste fold
    en analyse van gender bias via TPR Gap
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(skf.split(df, df["stratify_group"]))
    results = []
    detailed_results = []

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(df_):
        return tokenizer(
            df_["sentence"].tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
    
    # Stap 1: Hyperparameter optimalisatie op eerste fold
    logger.info(f"Stap 1: Hyperparameter optimalisatie op eerste fold voor {model_name}")
    
    train_idx, test_idx = folds[0]
    train_df_full = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Inner stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_indices, val_indices = next(
        sss.split(train_df_full, train_df_full["stratify_group"])
    )
    train_df = train_df_full.iloc[train_indices]
    val_df = train_df_full.iloc[val_indices]

    # Tokenize
    train_enc = tokenize(train_df)
    val_enc = tokenize(val_df)

    # Create datasets
    train_ds = TorchDataset(train_enc, train_df["stereotype"])
    val_ds = TorchDataset(val_enc, val_df["stereotype"])

    # Model init
    model_init = create_model_init(model_name)

    # Base TrainingArguments
    output_base = f"./results/{model_name.replace('/', '_')}/fold_hp_search"
    log_base = f"./logs/{model_name.replace('/', '_')}/fold_hp_search"

    training_args = TrainingArguments(
        output_dir=output_base,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir=log_base,
        report_to="none",
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_num_workers=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        seed=SEED,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    # Hyperparameterruimte
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [8, 16, 32]
            ),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        },
        n_trials=n_trials,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    
    # Beste hyperparameters weergeven en opslaan
    best_hyperparams = best_run.hyperparameters
    logger.info(f"Beste hyperparameters: {best_hyperparams}")
    
    # Stap 2: Alle folds trainen met beste hyperparameters
    logger.info(f"Stap 2: Alle folds trainen met beste hyperparameters en gender bias meten")
    
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        logger.info(f"Fold {fold}/{n_splits} voor {model_name}")

        train_df_full = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Inner stratified split voor validatie
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_indices, val_indices = next(
            sss.split(train_df_full, train_df_full["stratify_group"])
        )
        train_df = train_df_full.iloc[train_indices]
        val_df = train_df_full.iloc[val_indices]

        # Tokenize
        train_enc = tokenize(train_df)
        val_enc = tokenize(val_df)
        test_enc = tokenize(test_df)

        # Create datasets
        train_ds = TorchDataset(train_enc, train_df["stereotype"])
        val_ds = TorchDataset(val_enc, val_df["stereotype"])
        test_ds = TorchDataset(test_enc, test_df["stereotype"])

        # Model init
        model_init = create_model_init(model_name)

        # Training arguments met beste hyperparameters
        output_base = f"./results/{model_name.replace('/', '_')}/fold_{fold}"
        log_base = f"./logs/{model_name.replace('/', '_')}/fold_{fold}"

        training_args = TrainingArguments(
            output_dir=output_base,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            logging_dir=log_base,
            report_to="none",
            num_train_epochs=5,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            dataloader_num_workers=4,
            per_device_train_batch_size=best_hyperparams["per_device_train_batch_size"],
            per_device_eval_batch_size=32,
            learning_rate=best_hyperparams["learning_rate"],
            weight_decay=best_hyperparams["weight_decay"],
            warmup_ratio=best_hyperparams.get("warmup_ratio", 0.0),
            seed=SEED,
        )

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Training met beste hyperparameters
        trainer.train()

        # Standaard test evaluatie
        metrics = trainer.evaluate(eval_dataset=test_ds)
        
        # Gedetailleerde metrics inclusief TPR Gap
        # We slaan voorspellingen op om gedetailleerde metrics te berekenen
        test_output = trainer.predict(test_dataset=test_ds)
        detailed_metrics = compute_detailed_metrics(test_output, test_df)
        
        logger.info(
            f"Test metrics fold {fold} – "
            f"Acc: {metrics['eval_accuracy']:.4f} | "
            f"F1: {metrics['eval_f1']:.4f} | "
            f"TPR Gap (S): {detailed_metrics.get('tpr_gap_s', 'N/A'):.4f} | "
            f"TPR Gap (Contra-S): {detailed_metrics.get('tpr_gap_contra', 'N/A'):.4f}"
        )
        
        # Voeg metadata toe voor betere analyse
        metrics['fold'] = fold
        metrics['model'] = model_name
        metrics['hyperparams'] = best_hyperparams
        results.append(metrics)
        
        # Gedetailleerde resultaten met stratificatie per model en temperatuur
        for model_type in test_df['model'].unique():
            for temp in test_df['temperature'].unique():
                subset = test_df[(test_df['model'] == model_type) & 
                                (test_df['temperature'] == temp)]
                
                if len(subset) < 10:  # Skip als er te weinig data is
                    continue
                    
                subset_indices = subset.index
                subset_preds = np.argmax(test_output.predictions[subset.index.to_numpy() - min(test_df.index)], axis=1)
                subset_labels = test_output.label_ids[subset.index.to_numpy() - min(test_df.index)]
                
                # Bereken metrics voor deze specifieke subset
                acc = accuracy_score(subset_labels, subset_preds)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    subset_labels, subset_preds, average='binary', zero_division=0
                )
                
                # Bereken TPR Gap voor deze subset
                subset_detailed = compute_detailed_metrics(
                    type('obj', (object,), {
                        "predictions": test_output.predictions[subset.index.to_numpy() - min(test_df.index)],
                        "label_ids": test_output.label_ids[subset.index.to_numpy() - min(test_df.index)]
                    }), 
                    subset
                )
                
                detailed_result = {
                    'fold': fold,
                    'classifier_model': model_name,
                    'llm_model': model_type,
                    'temperature': temp,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'tpr_gap_s': subset_detailed.get('tpr_gap_s', None),
                    'tpr_gap_contra': subset_detailed.get('tpr_gap_contra', None),
                    'sample_size': len(subset)
                }
                
                detailed_results.append(detailed_result)

    # Resultaten samenvatten en opslaan
    results_df = pd.DataFrame(results)
    detailed_results_df = pd.DataFrame(detailed_results)
    
    # Algemene resultaten samenvatten
    avg_metrics = {
        'avg_accuracy': results_df['eval_accuracy'].mean(),
        'std_accuracy': results_df['eval_accuracy'].std(),
        'avg_f1': results_df['eval_f1'].mean(),
        'std_f1': results_df['eval_f1'].std(),
        'model': model_name,
        'hyperparams': best_hyperparams
    }
    
    # Log gemiddelde resultaten
    logger.info(f"Gemiddelde resultaten voor {model_name}:")
    logger.info(f"Accuracy: {avg_metrics['avg_accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
    logger.info(f"F1-score: {avg_metrics['avg_f1']:.4f} ± {avg_metrics['std_f1']:.4f}")
    
    # Diepere analyse per model en temperatuur
    grouped_results = detailed_results_df.groupby(['llm_model', 'temperature']).agg({
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'tpr_gap_s': ['mean', 'std'],
        'tpr_gap_contra': ['mean', 'std'],
        'sample_size': 'mean'
    }).reset_index()
    
    # Sla alle resultaten op
    results_df.to_csv(f"results_base_{model_name.replace('/', '_')}.csv", index=False)
    detailed_results_df.to_csv(f"results_detailed_{model_name.replace('/', '_')}.csv", index=False)
    grouped_results.to_csv(f"results_by_model_temp_{model_name.replace('/', '_')}.csv", index=False)
    pd.DataFrame([avg_metrics]).to_csv(f"results_summary_{model_name.replace('/', '_')}.csv", index=False)
    
    return results_df, detailed_results_df, grouped_results, avg_metrics

# ---------------------------------------------------------------------------
# Borda Count Rank Aggregation voor Bias Ranking -------------------------
# ---------------------------------------------------------------------------

def borda_count_ranking(dataframe, bias_columns=['tpr_gap_s_abs', 'tpr_gap_contra_abs']):
    """
    Implementeert Borda Count Ranking voor het aggregeren van TPR Gap ranks
    """
    # Absolute waarde van de TPR gap metrics maken
    if 'tpr_gap_s' in dataframe.columns and 'tpr_gap_s_abs' not in dataframe.columns:
        dataframe['tpr_gap_s_abs'] = dataframe['tpr_gap_s'].abs()
    if 'tpr_gap_contra' in dataframe.columns and 'tpr_gap_contra_abs' not in dataframe.columns:
        dataframe['tpr_gap_contra_abs'] = dataframe['tpr_gap_contra'].abs()
    
    # Rank per bias metric (lagere bias = betere rank)
    ranks = {}
    for col in bias_columns:
        if col in dataframe.columns:
            # Sorteer op basis van bias waarde (lager = beter)
            ranks[col] = dataframe.sort_values(by=col).reset_index()['llm_model'].tolist()
    
    # Als we geen valid ranks hebben
    if not ranks:
        return None
    
    # Bereken Borda scores (hogere score = betere rank)
    n = len(dataframe)
    borda_scores = {}
    
    for model in dataframe['llm_model'].unique():
        borda_scores[model] = 0
        for col, ranked_list in ranks.items():
            if model in ranked_list:
                # Geef punten gebaseerd op rank (n - rank)
                rank = ranked_list.index(model)
                borda_scores[model] += (n - rank)
    
    # Sorteer op basis van Borda score (hoger = beter)
    ranked_models = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Maak een dataframe van de resultaten
    rank_df = pd.DataFrame({
        'llm_model': [x[0] for x in ranked_models],
        'borda_score': [x[1] for x in ranked_models],
        'rank': range(1, len(ranked_models) + 1)
    })
    
    return rank_df

# ---------------------------------------------------------------------------
# Visualisatie functies ---------------------------------------------------
# ---------------------------------------------------------------------------

def visualize_tpr_gaps(grouped_results, output_file="tpr_gap_visualization.png"):
    """
    Visualiseert de TPR Gap per model en temperatuur
    """
    plt.figure(figsize=(14, 8))
    
    # Zorg dat de kolommen juiste namen hebben als ze multi-index zijn
    if isinstance(grouped_results.columns, pd.MultiIndex):
        grouped_results = grouped_results.reset_index()
        # Platte kolommen maken
        grouped_results.columns = ['_'.join(col).strip('_') for col in grouped_results.columns.values]
    
    # Voorbereiding data voor plot
    models = grouped_results['llm_model'].unique()
    temps = sorted(grouped_results['temperature'].unique())
    
    # Maak barplot
    bar_width = 0.35
    x = np.arange(len(models))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot voor stereotype TPR gap
    for i, temp in enumerate(temps):
        temp_data = grouped_results[grouped_results['temperature'] == temp]
        ax1.bar(x + i*bar_width - bar_width/2, 
                temp_data['tpr_gap_s_mean'], 
                bar_width, 
                label=f'Temp {temp}',
                yerr=temp_data['tpr_gap_s_std'])
    
    ax1.set_title('TPR Gap voor Stereotype Klasse (S)')
    ax1.set_xlabel('LLM Model')
    ax1.set_ylabel('TPR Gap (male - female)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.legend()
    
    # Plot voor contra-stereotype TPR gap
    for i, temp in enumerate(temps):
        temp_data = grouped_results[grouped_results['temperature'] == temp]
        ax2.bar(x + i*bar_width - bar_width/2, 
                temp_data['tpr_gap_contra_mean'], 
                bar_width, 
                label=f'Temp {temp}',
                yerr=temp_data['tpr_gap_contra_std'])
    
    ax2.set_title('TPR Gap voor Contra-Stereotype Klasse (¯S)')
    ax2.set_xlabel('LLM Model')
    ax2.set_ylabel('TPR Gap (male - female)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def visualize_accuracies(grouped_results, output_file="accuracy_visualization.png"):
    """
    Visualiseert de accuraatheid per model en temperatuur
    """
    plt.figure(figsize=(10, 6))
    
    # Zorg dat de kolommen juiste namen hebben als ze multi-index zijn
    if isinstance(grouped_results.columns, pd.MultiIndex):
        grouped_results = grouped_results.reset_index()
        # Platte kolommen maken
        grouped_results.columns = ['_'.join(col).strip('_') for col in grouped_results.columns.values]
    
    # Voorbereiding data voor plot
    models = grouped_results['llm_model'].unique()
    temps = sorted(grouped_results['temperature'].unique())
    
    # Maak barplot
    bar_width = 0.35
    x = np.arange(len(models))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot voor accuraatheid
    for i, temp in enumerate(temps):
        temp_data = grouped_results[grouped_results['temperature'] == temp]
        ax.bar(x + i*bar_width - bar_width/2, 
               temp_data['accuracy_mean'], 
               bar_width, 
               label=f'Temp {temp}',
               yerr=temp_data['accuracy_std'])
    
    ax.set_title('Classification Accuracy per Model en Temperatuur')
    ax.set_xlabel('LLM Model')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.5, 1.0)  # Typische accuracy range
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# ---------------------------------------------------------------------------
# Main -------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    # Nederlandse taalmodellen voor classificatie
    models = [
        ("GroNLP/bert-base-dutch-cased", "GroNLP/bert-base-dutch-cased"),
        ("bert-base-multilingual-cased", "bert-base-multilingual-cased"),
        ("DTAI-KULeuven/robbert-2023-dutch-large", "DTAI-KULeuven/robbert-2023-dutch-large"),
    ]

    # Resultaten opslaan per model
    all_summary_metrics = []
    all_detailed_results_dfs = []
    all_grouped_results = []
    
    # Voor elk model
    for model_name, tokenizer_name in models:
        logger.info(f"Starting model {model_name}")
        _, detailed_results_df, grouped_results, summary_metrics = run_gender_bias_cv(
            model_name, tokenizer_name, df, n_splits=5, n_trials=10
        )
        all_summary_metrics.append(summary_metrics)
        all_detailed_results_dfs.append(detailed_results_df)
        all_grouped_results.append(grouped_results)
    
    # Alle resultaten combineren en vergelijken
    summary_df = pd.DataFrame(all_summary_metrics)
    summary_df.to_csv("all_models_comparison.csv", index=False)
    
    # Combineer alle gedetailleerde resultaten
    combined_detailed = pd.concat(all_detailed_results_dfs)
    combined_detailed.to_csv("all_detailed_results.csv", index=False)
    
    # Gemiddelden per LLM model en temperatuur over alle classificatie modellen
    model_temp_summary = combined_detailed.groupby(['llm_model', 'temperature']).agg({
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'tpr_gap_s': ['mean', 'std', lambda x: abs(x).mean()],
        'tpr_gap_contra': ['mean', 'std', lambda x: abs(x).mean()],
        'sample_size': 'mean'
    })
    
    # Hernoem aangepaste functies
    model_temp_summary.columns = ['_'.join(col).strip('_') for col in model_temp_summary.columns.values]
    model_temp_summary = model_temp_summary.rename(columns={
        'tpr_gap_s_<lambda>': 'tpr_gap_s_abs',
        'tpr_gap_contra_<lambda>': 'tpr_gap_contra_abs'
    })
    
    # Voer Borda Count Ranking uit
    borda_ranks = borda_count_ranking(model_temp_summary.reset_index())
    borda_ranks.to_csv("bias_ranking_borda.csv", index=False)
    
    # Visualisaties
    visualize_tpr_gaps(model_temp_summary.reset_index(), "tpr_gap_by_model_temp.png")
    visualize_accuracies(model_temp_summary.reset_index(), "accuracy_by_model_temp.png")
    
    # Print vergelijking van alle modellen
    print("\nVergelijking van alle classificatiemodellen:")
    for _, row in summary_df.iterrows():
        print(f"{row['model']}:")
        print(f"  Accuracy: {row['avg_accuracy']:.4f} ± {row['std_accuracy']:.4f}")
        print(f"  F1-score: {row['avg_f1']:.4f} ± {row['std_f1']:.4f}")
        print()
    
    print("\nBias ranking (Borda Count) van LLM modellen:")
    for _, row in borda_ranks.iterrows():
        print(f"Rank {row['rank']}: {row['llm_model']} (Score: {row['borda_score']})")