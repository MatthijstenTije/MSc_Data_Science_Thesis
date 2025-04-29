# gender_summary.py
import logging
from pathlib import Path
import pandas as pd
from config import OUTPUT_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("gender_summary")

def generate_gender_summary(input_path=Path(OUTPUT_FOLDER) / "sentences_final.csv", encoding="utf-8"):
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
    summary.to_csv(Path(OUTPUT_FOLDER) / "summary_by_gender.csv" , index=False, encoding=encoding)
    logger.info(f"Saved summary breakdown (with percentages) to {Path(OUTPUT_FOLDER) / "summary_by_gender.csv"}")
    
    return summary

if __name__ == "__main__":
    generate_gender_summary()