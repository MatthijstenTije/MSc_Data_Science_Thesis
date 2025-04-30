from pathlib import Path
import pandas as pd

# 1) point to your summary folder
summary_dir = Path("phase_03") / "results" / "summary"

# 2) glob in all the per‐folder tpr_gap summary CSVs
files = list(summary_dir.glob("*_tpr_gap_summary.csv"))
if not files:
    raise FileNotFoundError(f"No *_tpr_gap_summary.csv found in {summary_dir}")

# 3) read & extract model name from filename
dfs = []
for fp in files:
    df = pd.read_csv(fp)
    
    # Extract model name directly from the filename
    model_name = fp.stem.replace("_tpr_gap_summary", "")
    
    # Add model name from file as a column
    df["Model_Name"] = model_name
    dfs.append(df)

all_models = pd.concat(dfs, ignore_index=True)

# 4) sort by Model and Model_Name (or by any columns you like)
all_models = all_models.sort_values(["Model", "Model_Name"]).reset_index(drop=True)

# 5a) save combined CSV
out_csv = summary_dir / "all_models_tpr_gap_summary.csv"
all_models.to_csv(out_csv, index=False)
print("✔️ Saved combined CSV →", out_csv)
print("\n### Preview:\n")
print(all_models.to_markdown(index=False))