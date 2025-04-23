import os
import json
import glob
import pandas as pd

# Base path where your logs are stored
logs_base_dir = "Notebooks/Phase_02/logs"

# Grab all JSONL log files in model‑specific subdirectories
log_files = glob.glob(os.path.join(logs_base_dir, "*", "*.jsonl"))

# Prepare a list to collect rows
rows = []

# Loop over each log file
for log_file in log_files:
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log_data = json.loads(line)
                model             = log_data.get("model")
                temperature       = log_data.get("temperature")
                noun_gender       = log_data.get("noun_gender")
                adjective_gender  = log_data.get("adjective_gender")
                total_runs        = log_data.get("total_runs")
                sentences         = log_data.get("aggregated_output", [])

                for entry in sentences:
                    word          = entry.get("word")
                    sentence      = entry.get("sentence")

                    if sentence:
                        rows.append({
                            "word":             word,
                            "sentence":         sentence,
                            "model":            model,
                            "noun_gender":      noun_gender,
                            "adjective_gender": adjective_gender,
                            "temperature":      temperature,
                            "total_runs":       total_runs,
                        })
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON in file: {log_file}")
                continue

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save to CSV
out_path = "Notebooks/Phase_02/output/aggregated_sentences.csv"
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"Finished processing. Data saved to '{out_path}'")
