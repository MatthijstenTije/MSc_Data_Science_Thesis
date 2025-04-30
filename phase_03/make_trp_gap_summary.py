from pathlib import Path
import pandas as pd

# --- Setup paths ---
base_dir = Path("phase_03") / "results"
summary_dir = base_dir / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)

# --- Store folder names and their CSVs ---
folder_files = {}  # maps folder name → list of matching CSVs

for folder in base_dir.iterdir():
    if not folder.is_dir() or folder.name == "summary":
        continue
    csvs = list(folder.glob("results_by_model_temp_*.csv"))
    if csvs:
        folder_files[folder.name] = csvs

if not folder_files:
    raise FileNotFoundError("No matching CSVs found in any folders.")

# --- Process each file ---
for folder_name, files in folder_files.items():
    for fn in files:
        df = pd.read_csv(fn, header=[0, 1])
        flat_cols = []
        for c0, c1 in df.columns:
            if pd.isna(c1) or str(c1).strip() == "":
                flat_cols.append(c0)
            else:
                flat_cols.append(f"{c0}_{c1}")
        df.columns = flat_cols

        summary_rows = []
        for model, grp in df.groupby("llm_model_Unnamed: 0_level_1"):
            n = grp["sample_size_mean"]
            w_s = (grp["tpr_gap_s_mean"] * n).sum() / n.sum()
            w_con = (grp["tpr_gap_contra_mean"] * n).sum() / n.sum()
            total_n = n.sum()
            summary_rows.append({
                "Model": model,
                "TPR Gap (S)": w_s,
                "TPR Gap (Contra)": w_con,
                "Total N": int(total_n)
            })

        file_summary = pd.DataFrame(summary_rows).sort_values("Model")

        # Use folder and filename to make a unique basename
        name_prefix = f"{folder_name}"
        csv_path = summary_dir / f"{name_prefix}_tpr_gap_summary.csv"

        file_summary.to_csv(csv_path, index=False)
        print(f"\n--- Summary for {csv_path} ---")
        print(file_summary.to_markdown(index=False))
        print(f"✔︎ CSV → {csv_path}")

print("\nAll per-file summaries written to:", summary_dir)
