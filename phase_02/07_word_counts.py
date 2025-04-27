# word_counts.py
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("word_counts")

def analyze_word_counts(input_path="Notebooks/Phase_02/output/sentences_cleaned.csv", encoding="utf-8"):
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
    output_path = Path("Notebooks/Phase_02/output/word_counts.csv")
    word_counts.to_csv(output_path, index=False, encoding=encoding)
    logger.info(f"Saved word counts to {output_path}")
    
    return word_counts

if __name__ == "__main__":
    analyze_word_counts()