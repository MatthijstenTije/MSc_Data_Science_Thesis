# model_gender_summary.py
import logging
from pathlib import Path
import pandas as pd
from config import OUTPUT_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("model_gender_summary")

def generate_model_gender_summary(input_path=Path(OUTPUT_FOLDER) / "sentences_final.csv", encoding="utf-8"):
    """
    Generate and save summary by model × noun_gender × adjective_gender with per-model percentages.
    """
    # Load the data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, encoding=encoding)
    
    # Summary by model × noun_gender × adjective_gender
    summary = (
        df
        .groupby(['model', 'noun_gender', 'adjective_gender'])
        .size()
        .reset_index(name='sentence_count')
    )
    
    # Compute percentage within each model
    summary['percentage'] = (
        summary
        .groupby('model')['sentence_count']
        .transform(lambda x: x / x.sum() * 100)
    )
    
    # Sort for readability
    summary = summary.sort_values(['model', 'sentence_count'], 
                                 ascending=[True, False])
    
    # Display the summary
    logger.info("Sentence counts by model, noun_gender & adjective_gender (out of {:,} sentences)"
                .format(len(df)))
    logger.info("\n" + summary.to_string(index=False))
    
    # Save to disk
    summary.to_csv(Path(OUTPUT_FOLDER) / "summary_model_gender.csv", index=False, encoding=encoding)
    logger.info(f"Saved summary to {Path(OUTPUT_FOLDER) / "summary_model_gender.csv"}")
    
    return summary

if __name__ == "__main__":
    generate_model_gender_summary()