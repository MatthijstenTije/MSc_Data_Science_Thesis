from pathlib import Path
import pandas as pd

# 1) Base results folder (relative)
base_dir = Path("phase_03/results")

# 2) Make a subfolder for summaries
summary_dir = base_dir / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)

# 3) Find all the comparison CSVs
files = list(base_dir.rglob("all_models_comparison*.csv"))

# 4) Read & extract only the model + accuracy cols
dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df[['model', 'avg_accuracy', 'std_accuracy']])

# 5) Combine & sort by descending mean accuracy
combined = (
    pd.concat(dfs, ignore_index=True)
      .sort_values('avg_accuracy', ascending=False)
      .reset_index(drop=True)
)

# 6a) Save as CSV
csv_path = summary_dir / "model_accuracy_summary.csv"
combined.to_csv(csv_path, index=False)

print(f"✔️ Saved summary CSV → {csv_path}")