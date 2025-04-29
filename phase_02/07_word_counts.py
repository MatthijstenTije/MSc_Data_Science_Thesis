# word_counts.py
import logging
from pathlib import Path
import pandas as pd
from config import OUTPUT_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("word_counts")

def analyze_word_counts(input_path=Path(OUTPUT_FOLDER) / "sentences_final.csv", encoding="utf-8"):
    """
    Analyze word occurrences in the dataset.
    """
    # Load the data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, encoding=encoding)
    
    # Total number of rows per word
    word_counts = (
        df['word']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'word', 'word': 'occurrence_count'})
    )
    
    # Display and save results
    logger.info("Word occurrence counts (top 20):")
    logger.info("\n" + str(word_counts.head(20)))
    
    # Save to CSV
    word_counts.to_csv(Path(OUTPUT_FOLDER) / "word_counts.csv", index=False, encoding=encoding)
    logger.info(f"Saved word counts to {Path(OUTPUT_FOLDER) / "word_counts.csv"}")
    
    return word_counts

if __name__ == "__main__":
    analyze_word_counts()