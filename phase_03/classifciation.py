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
    df["stereotype"] = df["label"].isin(["MM", "FF"]).astype(int)
    
    # These columns are only for post-hoc analysis, not for the classifier
    df["stereotype_type"] = df.apply(
        lambda r: "S" if r["stereotype"] == 1 else "S_bar", 
        axis=1
    )
    df["gender"] = df.apply(
        lambda r: "male" if r.noun_gender == "male" else "female",
        axis=1,
    )
    
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

def compute_tpr_gap(y_true, y_pred, gender_data):
    """
    Calculate TPRgap (True Positive Rate Gap) between male and female groups
    TPRgap = TPRmale - TPRfemale
    """
    # Extract male and female data points
    male_indices = [i for i, g in enumerate(gender_data) if g == "male"]
    female_indices = [i for i, g in enumerate(gender_data) if g == "female"]
    
    # Calculate TPRs for each gender
    if len(male_indices) > 0:
        male_y_true = [y_true[i] for i in male_indices]
        male_y_pred = [y_pred[i] for i in male_indices]
        tn_m, fp_m, fn_m, tp_m = confusion_matrix(male_y_true, male_y_pred, labels=[0, 1]).ravel()
        tpr_male = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
    else:
        tpr_male = 0

    if len(female_indices) > 0:
        female_y_true = [y_true[i] for i in female_indices]
        female_y_pred = [y_pred[i] for i in female_indices]
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(female_y_true, female_y_pred, labels=[0, 1]).ravel()
        tpr_female = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
    else:
        tpr_female = 0
    
    tpr_gap = tpr_male - tpr_female
    return tpr_gap, tpr_male, tpr_female

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
# Training and evaluation -------------------------------------------------
# ---------------------------------------------------------------------------

def run_cv_hp_search(model_name, tokenizer_name, df, n_splits=5, n_trials=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = []
    detailed_results = []  # For storing detailed bias metrics
    all_sentence_predictions = []  # For storing sentence-level predictions

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(df_):
        # Only use sentence text for tokenization - no gender info!
        return tokenizer(
            df_["sentence"].tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(df, df["stratify_group"]), start=1
    ):
        logger.info(f"Fold {fold}/{n_splits} for {model_name}")

        train_df_full = df.iloc[train_idx]
        test_df = df.iloc[test_idx].reset_index(drop=True)  # Reset index for easy tracking

        # Inner stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_indices, val_indices = next(
            sss.split(train_df_full, train_df_full["stratify_group"])
        )
        train_df = train_df_full.iloc[train_indices]
        val_df = train_df_full.iloc[val_indices]

        # Tokenize - only sentence text is used here
        train_enc = tokenize(train_df)
        val_enc = tokenize(val_df)
        test_enc = tokenize(test_df)

        # Create datasets - only uses text and stereotype label (0/1)
        train_ds = TorchDataset(train_enc, train_df["stereotype"])
        val_ds = TorchDataset(val_enc, val_df["stereotype"])
        test_ds = TorchDataset(test_enc, test_df["stereotype"])

        # Model init
        model_init = create_model_init(model_name)

        # Base TrainingArguments
        output_base = f"./results/{model_name.replace('/', '_')}/fold_{fold}"
        log_base = f"./logs/{model_name.replace('/', '_')}/fold_{fold}"

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

        # Hyperparameter search
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
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        logger.info(f"Best hyperparameters fold {fold}: {best_run.hyperparameters}")

        # Apply best hyperparameters
        trainer.args.per_device_train_batch_size = best_run.hyperparameters["per_device_train_batch_size"]
        trainer.args.learning_rate = best_run.hyperparameters["learning_rate"]
        trainer.args.weight_decay = best_run.hyperparameters.get("weight_decay", 0.0)

        # Retrain
        trainer.train()

        # Test evaluation
        metrics = trainer.evaluate(eval_dataset=test_ds)
        logger.info(
            f"Test metrics fold {fold} – "
            f"Acc: {metrics['eval_accuracy']:.4f} | "
            f"Prec: {metrics['eval_precision']:.4f} | "
            f"Rec: {metrics['eval_recall']:.4f} | "
            f"F1: {metrics['eval_f1']:.4f}"
        )
        results.append(metrics)
        
        # Get predictions for gender bias analysis
        pred_output = trainer.predict(test_ds)
        preds = np.argmax(pred_output.predictions, axis=1)
        probas = torch.nn.functional.softmax(torch.tensor(pred_output.predictions), dim=1).numpy()
        
        # ------- STORE SENTENCE-LEVEL PREDICTIONS -------
        # Create a dataframe with all test sentences and their predictions
        sentences_with_preds = test_df.copy()
        sentences_with_preds["fold"] = fold
        sentences_with_preds["classifier_model"] = model_name
        sentences_with_preds["predicted_stereotype"] = preds
        sentences_with_preds["proba_stereotype"] = probas[:, 1]  # Probability of stereotype class (1)
        sentences_with_preds["proba_non_stereotype"] = probas[:, 0]  # Probability of non-stereotype class (0)
        sentences_with_preds["correct_prediction"] = sentences_with_preds["stereotype"] == sentences_with_preds["predicted_stereotype"]
        
        # Add to our collection of all sentence predictions
        all_sentence_predictions.append(sentences_with_preds)
        
        # ------- BIAS ANALYSIS (POST-HOC, NO LEAKAGE) -------
        # Calculate TPRgap for overall dataset
        overall_tpr_gap, tpr_male, tpr_female = compute_tpr_gap(
            test_df["stereotype"].tolist(), 
            preds, 
            test_df["gender"].tolist()
        )
        
        # Calculate TPRgap for stereotype-consistent examples (S)
        s_df = test_df[test_df["stereotype_type"] == "S"]
        if not s_df.empty:
            s_indices = s_df.index
            s_preds = [preds[i] for i in s_indices]
            s_tpr_gap, s_tpr_male, s_tpr_female = compute_tpr_gap(
                s_df["stereotype"].tolist(),
                s_preds,
                s_df["gender"].tolist()
            )
        else:
            s_tpr_gap, s_tpr_male, s_tpr_female = 0, 0, 0
            
        # Calculate TPRgap for stereotype-contradictory examples (S_bar)
        s_bar_df = test_df[test_df["stereotype_type"] == "S_bar"]
        if not s_bar_df.empty:
            s_bar_indices = s_bar_df.index
            s_bar_preds = [preds[i] for i in s_bar_indices]
            s_bar_tpr_gap, s_bar_tpr_male, s_bar_tpr_female = compute_tpr_gap(
                s_bar_df["stereotype"].tolist(),
                s_bar_preds,
                s_bar_df["gender"].tolist()
            )
        else:
            s_bar_tpr_gap, s_bar_tpr_male, s_bar_tpr_female = 0, 0, 0
        
        # Save detailed metrics per model and fold
        fold_metrics = {
            "model": model_name,
            "fold": fold,
            "accuracy": metrics["eval_accuracy"],
            "precision": metrics["eval_precision"],
            "recall": metrics["eval_recall"],
            "f1": metrics["eval_f1"],
            "overall_tpr_gap": overall_tpr_gap,
            "overall_tpr_male": tpr_male,
            "overall_tpr_female": tpr_female,
            "stereotype_tpr_gap": s_tpr_gap,
            "stereotype_tpr_male": s_tpr_male,
            "stereotype_tpr_female": s_tpr_female,
            "contradict_tpr_gap": s_bar_tpr_gap,
            "contradict_tpr_male": s_bar_tpr_male,
            "contradict_tpr_female": s_bar_tpr_female,
        }
        detailed_results.append(fold_metrics)
        
        # Also analyze at model-temperature level
        # Group test data by temperature and repeat analysis
        for temp, temp_group in test_df.groupby("temperature"):
            # Get predictions for this temperature group
            temp_indices = temp_group.index
            temp_preds = [preds[i] for i in temp_indices]
            
            # Overall TPR gap for this temperature
            temp_tpr_gap, temp_tpr_male, temp_tpr_female = compute_tpr_gap(
                temp_group["stereotype"].tolist(),
                temp_preds,
                temp_group["gender"].tolist()
            )
            
            # TPR gap for stereotype-consistent at this temperature
            temp_s_df = temp_group[temp_group["stereotype_type"] == "S"]
            if not temp_s_df.empty:
                temp_s_indices = temp_s_df.index
                temp_s_preds = [preds[i] for i in temp_s_indices]
                temp_s_tpr_gap, temp_s_tpr_male, temp_s_tpr_female = compute_tpr_gap(
                    temp_s_df["stereotype"].tolist(),
                    temp_s_preds,
                    temp_s_df["gender"].tolist()
                )
            else:
                temp_s_tpr_gap, temp_s_tpr_male, temp_s_tpr_female = 0, 0, 0
                
            # TPR gap for stereotype-contradictory at this temperature
            temp_s_bar_df = temp_group[temp_group["stereotype_type"] == "S_bar"]
            if not temp_s_bar_df.empty:
                temp_s_bar_indices = temp_s_bar_df.index
                temp_s_bar_preds = [preds[i] for i in temp_s_bar_indices]
                temp_s_bar_tpr_gap, temp_s_bar_tpr_male, temp_s_bar_tpr_female = compute_tpr_gap(
                    temp_s_bar_df["stereotype"].tolist(),
                    temp_s_bar_preds,
                    temp_s_bar_df["gender"].tolist()
                )
            else:
                temp_s_bar_tpr_gap, temp_s_bar_tpr_male, temp_s_bar_tpr_female = 0, 0, 0
            
            # Save temperature-level metrics
            temp_metrics = {
                "model": model_name,
                "fold": fold,
                "temperature": temp,
                "accuracy": accuracy_score(temp_group["stereotype"].tolist(), temp_preds),
                "overall_tpr_gap": temp_tpr_gap,
                "overall_tpr_male": temp_tpr_male, 
                "overall_tpr_female": temp_tpr_female,
                "stereotype_tpr_gap": temp_s_tpr_gap,
                "stereotype_tpr_male": temp_s_tpr_male,
                "stereotype_tpr_female": temp_s_tpr_female,
                "contradict_tpr_gap": temp_s_bar_tpr_gap,
                "contradict_tpr_male": temp_s_bar_tpr_male,
                "contradict_tpr_female": temp_s_bar_tpr_female,
            }
            detailed_results.append(temp_metrics)

    # Save results
    pd.DataFrame(results).to_csv(f"results_{model_name.replace('/', '_')}.csv", index=False)
    pd.DataFrame(detailed_results).to_csv(f"detailed_results_{model_name.replace('/', '_')}.csv", index=False)
    
    # Combine and save all sentence predictions
    all_sentences_df = pd.concat(all_sentence_predictions, ignore_index=True)
    all_sentences_df.to_csv(f"sentence_predictions_{model_name.replace('/', '_')}.csv", index=False)
    
    return results, detailed_results, all_sentences_df


# ---------------------------------------------------------------------------
# Analysis functions ------------------------------------------------------
# ---------------------------------------------------------------------------

def generate_bias_summary(detailed_results_all_models):
    """Generate a summary of gender bias across all models"""
    summary_data = []
    
    # Group by model and calculate average metrics
    model_groups = pd.DataFrame(detailed_results_all_models).groupby("model")
    for model, model_data in model_groups:
        # Filter to only include fold-level metrics (not temperature level)
        fold_data = model_data[model_data["temperature"].isna()]
        
        # Calculate averages across folds
        avg_metrics = {
            "model": model,
            "accuracy": fold_data["accuracy"].mean(),
            "stereotype_tpr_gap": fold_data["stereotype_tpr_gap"].mean(),
            "contradict_tpr_gap": fold_data["contradict_tpr_gap"].mean(),
            "overall_tpr_gap": fold_data["overall_tpr_gap"].mean(),
        }
        summary_data.append(avg_metrics)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Save and return
    summary_df.to_csv("bias_summary_by_model.csv", index=False)
    return summary_df

def generate_temperature_bias_summary(detailed_results_all_models):
    """Generate a summary of gender bias across models and temperatures"""
    summary_data = []
    
    # Convert to DataFrame if it's a list
    if isinstance(detailed_results_all_models, list):
        detailed_results_df = pd.DataFrame(detailed_results_all_models)
    else:
        detailed_results_df = detailed_results_all_models
    
    # Group by model and temperature and calculate average metrics
    for (model, temp), group_data in detailed_results_df.groupby(["model", "temperature"]):
        # Skip fold-level entries (which have no temperature)
        if pd.isna(temp):
            continue
            
        # Calculate averages across folds for this model+temperature
        avg_metrics = {
            "model": model,
            "temperature": temp,
            "accuracy": group_data["accuracy"].mean(),
            "stereotype_tpr_gap": group_data["stereotype_tpr_gap"].mean(),
            "contradict_tpr_gap": group_data["contradict_tpr_gap"].mean(),
            "overall_tpr_gap": group_data["overall_tpr_gap"].mean(),
        }
        summary_data.append(avg_metrics)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Save and return
    summary_df.to_csv("bias_summary_by_model_temperature.csv", index=False)
    return summary_df

def apply_borda_ranking(summary_df):
    """
    Apply Borda count rank aggregation to rank models by bias
    Lower absolute bias values are better (rank 1)
    """
    # Create a copy of the summary dataframe
    ranking_df = summary_df.copy()
    
    # Rank stereotype_tpr_gap by absolute value (ascending)
    ranking_df['stereotype_rank'] = ranking_df['stereotype_tpr_gap'].abs().rank()
    
    # Rank contradict_tpr_gap by absolute value (ascending)
    ranking_df['contradict_rank'] = ranking_df['contradict_tpr_gap'].abs().rank()
    
    # Calculate Borda score (sum of ranks)
    ranking_df['borda_score'] = ranking_df['stereotype_rank'] + ranking_df['contradict_rank']
    
    # Final ranking based on Borda score (lower is better)
    ranking_df['bias_rank'] = ranking_df['borda_score'].rank()
    
    # Save and return
    ranking_df.to_csv("model_bias_ranking.csv", index=False)
    return ranking_df

def create_bias_visualization(summary_df):
    """
    Create a more comprehensive summary table similar to the paper's Table 4
    """
    # Create a formatted table
    table_data = []
    for _, row in summary_df.iterrows():
        table_row = {
            "Model": row["model"].split("/")[-1],  # Just use the model name, not the full path
            "Accuracy (%)": f"{row['accuracy']*100:.1f}",
            "TPRgap in S": f"{row['stereotype_tpr_gap']:.2f}",
            "TPRgap in S̄": f"{row['contradict_tpr_gap']:.2f}",
            "Bias Rank": f"{int(row['bias_rank'])}"
        }
        table_data.append(table_row)
    
    result_table = pd.DataFrame(table_data)
    result_table.to_csv("bias_results_table.csv", index=False)
    return result_table

def analyze_sentence_examples(all_sentences_df, num_examples=5):
    """
    Extract interesting examples from the sentence predictions
    to better understand the classifier behavior
    """
    # Group examples by categories based on gender, stereotype and prediction correctness
    categories = []
    
    # Examples of stereotype sentences by gender
    for gender in ["male", "female"]:
        # Correctly classified stereotype sentences
        correct_stereo = all_sentences_df[
            (all_sentences_df["gender"] == gender) & 
            (all_sentences_df["stereotype"] == 1) & 
            (all_sentences_df["correct_prediction"] == True)
        ].sample(min(num_examples, len(all_sentences_df[
            (all_sentences_df["gender"] == gender) & 
            (all_sentences_df["stereotype"] == 1) & 
            (all_sentences_df["correct_prediction"] == True)
        ])))
        
        # Incorrectly classified stereotype sentences
        incorrect_stereo = all_sentences_df[
            (all_sentences_df["gender"] == gender) & 
            (all_sentences_df["stereotype"] == 1) & 
            (all_sentences_df["correct_prediction"] == False)
        ].sample(min(num_examples, len(all_sentences_df[
            (all_sentences_df["gender"] == gender) & 
            (all_sentences_df["stereotype"] == 1) & 
            (all_sentences_df["correct_prediction"] == False)
        ])))
        
        # Correctly classified non-stereotype sentences
        correct_non_stereo = all_sentences_df[
            (all_sentences_df["gender"] == gender) & 
            (all_sentences_df["stereotype"] == 0) & 
            (all_sentences_df["correct_prediction"] == True)
        ].sample(min(num_examples, len(all_sentences_df[
            (all_sentences_df["gender"] == gender) & 
            (all_sentences_df["stereotype"] == 0) & 
            (all_sentences_df["correct_prediction"] == True)
        ])))
        
        # Incorrectly classified non-stereotype sentences
        incorrect_non_stereo = all_sentences_df[
            (all_sentences_df["gender"] == gender) & 
            (all_sentences_df["stereotype"] == 0) & 
            (all_sentences_df["correct_prediction"] == False)
        ].sample(min(num_examples, len(all_sentences_df[
            (all_sentences_df["gender"] == gender) & 
            (all_sentences_df["stereotype"] == 0) & 
            (all_sentences_df["correct_prediction"] == False)
        ])))
        
        categories.extend([
            {"title": f"Correct {gender} stereotype sentences", "examples": correct_stereo},
            {"title": f"Incorrect {gender} stereotype sentences", "examples": incorrect_stereo},
            {"title": f"Correct {gender} non-stereotype sentences", "examples": correct_non_stereo},
            {"title": f"Incorrect {gender} non-stereotype sentences", "examples": incorrect_non_stereo},
        ])
    
    # Create a markdown file with examples
    with open("sentence_examples.md", "w") as f:
        f.write("# Sentence Classification Examples\n\n")
        
        for category in categories:
            f.write(f"## {category['title']}\n\n")
            
            for _, example in category["examples"].iterrows():
                f.write(f"- \"{example['sentence']}\"\n")
                f.write(f"  - True label: {'Stereotype' if example['stereotype'] == 1 else 'Non-stereotype'}\n")
                f.write(f"  - Predicted: {'Stereotype' if example['predicted_stereotype'] == 1 else 'Non-stereotype'}\n")
                f.write(f"  - Gender: {example['gender']}\n")
                f.write(f"  - Confidence: {example['proba_stereotype']:.3f}\n\n")
    
    return categories


# ---------------------------------------------------------------------------
# Main -------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    models = [
        ("GroNLP/bert-base-dutch-cased", "GroNLP/bert-base-dutch-cased")
        #("bert-base-multilingual-cased", "bert-base-multilingual-cased"),
        #("DTAI-KULeuven/robbert-2023-dutch-large", "DTAI-KULeuven/robbert-2023-dutch-large"),
    ]

    all_detailed_results = []
    all_sentences = []
    
    for model_name, tokenizer_name in models:
        logger.info(f"Starting model {model_name}")
        _, detailed_results, sentence_preds = run_cv_hp_search(model_name, tokenizer_name, df, n_splits=5, n_trials=5)
        all_detailed_results.extend(detailed_results)
        all_sentences.append(sentence_preds)
    
    # Combine all detailed results
    all_results_df = pd.DataFrame(all_detailed_results)
    
    # Combine all sentence predictions
    all_sentences_df = pd.concat(all_sentences, ignore_index=True)
    all_sentences_df.to_csv("all_sentence_predictions.csv", index=False)
    
    # Generate summaries
    model_summary = generate_bias_summary(all_results_df)
    print("Model Bias Summary:")
    print(model_summary)
    
    temp_summary = generate_temperature_bias_summary(all_results_df)
    print("\nModel-Temperature Bias Summary:")
    print(temp_summary)
    
    # Apply Borda ranking
    ranking = apply_borda_ranking(model_summary)
    print("\nModel Bias Ranking (lower rank = less bias):")
    print(ranking[['model', 'stereotype_tpr_gap', 'contradict_tpr_gap', 'bias_rank']])
    
    # Create final visualization
    result_table = create_bias_visualization(ranking)
    print("\nFinal Results Table:")
    print(result_table)
    
    # Analyze specific sentence examples
    print("\nExtracting sentence examples...")
    analyze_sentence_examples(all_sentences_df)
    print("Sentence examples saved to sentence_examples.md")