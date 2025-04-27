# gender_summary.py
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("gender_summary")

def generate_gender_summary(input_path="Notebooks/Phase_02/output/sentences_cleaned.csv", encoding="utf-8"):
    """
    Generate and save summary statistics by noun_gender × adjective_gender.
    """
    # Load the data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, encoding=encoding)
    
    # Calculate summary by noun_gender × adjective_gender (with percentages)
    total_sentences = len(df)
    summary = (
        df
        .groupby(['noun_gender', 'adjective_gender'])
        .size()
        .reset_index(name='sentence_count')
        .assign(
            percentage=lambda d: (d['sentence_count'] / total_sentences) * 100
        )
        .sort_values('sentence_count', ascending=False)
    )
    
    # Display summary
    logger.info("Sentence counts by noun_gender and adjective_gender (with % of total):")
    logger.info("\n" + str(summary))
    
    # Save summary to CSV
    summary_path = Path("Notebooks/Phase_02/output/summary_by_gender.csv")
    summary.to_csv(summary_path, index=False, encoding=encoding)
    logger.info(f"Saved summary breakdown (with percentages) to {summary_path}")
    
    return summary

if __name__ == "__main__":
    generate_gender_summary()