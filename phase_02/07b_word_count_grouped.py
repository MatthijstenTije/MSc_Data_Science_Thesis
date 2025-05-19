# grouped_word_counts_to_csv.py
import logging
from pathlib import Path
import pandas as pd
from config import OUTPUT_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("grouped_word_counts")

def analyze_grouped_word_counts(input_path=Path(OUTPUT_FOLDER) / "sentences_final.csv", encoding="utf-8"):
    """
    Analyze word (adjective) occurrences grouped by noun_gender and adjective_gender.
    """
    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, encoding=encoding)

    # Count occurrences grouped by word, noun_gender, and adjective_gender
    grouped_counts = (
        df.groupby(['word', 'noun_gender', 'adjective_gender'])
        .size()
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
    )

    # Save full grouped list to CSV
    output_csv = Path(OUTPUT_FOLDER) / "grouped_adjective_counts.csv"
    grouped_counts.to_csv(output_csv, index=False, encoding=encoding)
    logger.info(f"Saved grouped adjective counts to {output_csv}")

    return grouped_counts


if __name__ == "__main__":
    analyze_grouped_word_counts()
