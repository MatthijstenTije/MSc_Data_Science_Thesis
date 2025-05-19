# config.py
import os
from pathlib import Path 
from datetime import datetime

# === Base Directory ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Folders ===
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
PROMPT_FOLDER = os.path.join(BASE_DIR, "prompts")
VISUALIZATION_FOLDER = os.path.join(BASE_DIR, "visualizations")
INTERMEDIATE_FOLDER = os.path.join(OUTPUT_FOLDER, "intermediate")

# === Files ===
MAIN_LOG_PATH = os.path.join(LOG_FOLDER, "main.log")
QUALITY_LOG_PATH = os.path.join(LOG_FOLDER, "quality_issues.jsonl")
AGGREGATED_CSV_PATH = os.path.join(OUTPUT_FOLDER, "aggregated_sentences.csv")

# === Timestamps
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

# === Experiment Configurations ===
MODELS = [
    "llama3-chatqa:8b",
    "llama3:text",
    "llama3:8b",
    "llama2-uncensored"
]

TEMPERATURES = [0.5, 0.75, 1, 1.25, 1.5]
SEED_START = 42
MAX_BATCH_SIZE = 5
TARGET_COUNT_PER_WORD = 15
TOTAL_TARGET_SENTENCES = 200
MODEL_TIMEOUT = 600  # in seconds
SAVE_INTERVAL = 10  # how often to save intermediate results in minutes

# === Make sure all folders exist ===
for folder in [LOG_FOLDER, OUTPUT_FOLDER, PROMPT_FOLDER, VISUALIZATION_FOLDER, INTERMEDIATE_FOLDER]:
    os.makedirs(folder, exist_ok=True)