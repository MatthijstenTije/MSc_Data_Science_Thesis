import os
import json
import glob
import pandas as pd
from config import LOG_FOLDER, OUTPUT_FOLDER, AGGREGATED_CSV_PATH

# Grab all JSONL log files in model-specific subdirectories
log_files = glob.glob(os.path.join(LOG_FOLDER, "*", "*.jsonl"))

# Prepare a list to collect rows
rows = []

# Helper function to load JSON safely
def safe_load_json(file_path):
    """Tries to load as JSON array or as JSONL lines"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '{':
            # Full JSON file
            try:
                return [json.load(f)]
            except json.JSONDecodeError:
                return []
        else:
            # JSONL file
            return [json.loads(line) for line in f if line.strip()]

# Loop over each log file
for log_file in log_files:
    try:
        entries = safe_load_json(log_file)

        for log_data in entries:
            if not isinstance(log_data, dict):
                continue

            model             = log_data.get("model")
            temperature       = log_data.get("temperature")
            noun_gender       = log_data.get("noun_gender")
            adjective_gender  = log_data.get("adjective_gender")
            total_runs        = log_data.get("total_runs")
            sentences         = log_data.get("aggregated_output", [])

            for entry in sentences:
                if isinstance(entry, dict):
                    word     = entry.get("word")
                    sentence = entry.get("sentence")

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
    except Exception as e:
        print(f"Skipping file {log_file} because of error: {e}")

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save to CSV
df.to_csv(AGGREGATED_CSV_PATH, index=False, encoding="utf-8")

print(f"Finished processing. Data saved to '{AGGREGATED_CSV_PATH}'")
