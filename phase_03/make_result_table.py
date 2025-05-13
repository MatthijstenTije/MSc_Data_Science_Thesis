from pathlib import Path
import pandas as pd
import numpy as np

# 1) Base results folder (relative)
base_dir = Path("phase_03/results")

# 2) Make a subfolder for summaries
summary_dir = base_dir / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)

# 3) Find all the comparison CSVs
files = list(base_dir.rglob("all_detailed_result*.csv"))

# 4) Read & extract the model and all performance metrics
dfs = []
for f in files:
    df = pd.read_csv(f)
    print(df)
    metrics_df = df[['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1','model']]
    metrics_df = metrics_df.rename(columns={
        'eval_accuracy': 'accuracy',
        'eval_precision': 'precision',
        'eval_recall': 'recall',
        'eval_f1': 'f1'
    })
    
    # Add fold information if available
    if 'fold' in df.columns:
        metrics_df['fold'] = df['fold']
    
    dfs.append(metrics_df)

# 5) Combine all dataframes
combined = pd.concat(dfs, ignore_index=True)
print(combined)
# 6) Group by model to get mean and std for each metric
grouped = combined.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'precision': ['mean', 'std'],
    'recall': ['mean', 'std'],
    'f1': ['mean', 'std']
}).reset_index()

# 7) Flatten the column hierarchy
grouped.columns = [
    '_'.join(col).strip('_') for col in grouped.columns.values
]

# 8) Sort by descending mean accuracy
final_summary = grouped.sort_values('accuracy_mean', ascending=False).reset_index(drop=True)


# 10) Create a more readable version with rounded values
summary = final_summary.copy()
for col in summary.columns:
    if col != 'model' and '_mean' in col:
        summary[col] = summary[col].map(lambda x: f"{x:.4f}")
    elif col != 'model' and '_std' in col:
        summary[col] = summary[col].map(lambda x: f"±{x:.4f}")

# Alternate columns for readability
readable_cols = ['model']
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    readable_cols.extend([f'{metric}_mean', f'{metric}_std'])

summary = summary[readable_cols]

# 11) Save the readable summary
readable_path = summary_dir / "model_performance_summary_readable.csv"
summary.to_csv(readable_path, index=False)
print(f"✔️ Saved readable performance summary CSV → {readable_path}")