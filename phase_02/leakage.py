#!/usr/bin/env python3
"""
Adjective Analysis Script

This script analyzes adjectives in sentences, focusing on gender associations.
It processes labeled data to identify adjective distributions and gender stereotypes.

Key features:
- Load and validate sentence data from CSV
- Calculate leakage metrics (sentences with multiple adjectives)
- Analyze gender stereotypes in adjective usage
- Generate visualizations for analysis
- Manual curation of sentences to review analysis results
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import OUTPUT_FOLDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('adjective_analysis')

# Constants
ENCODING = 'utf-8'  # Define encoding for file operations
CSV_FINAL = Path(OUTPUT_FOLDER) / "sentences_final.csv"
CSV_REVIEW = Path(OUTPUT_FOLDER) / "sentences_to_review.csv"
CSV_CURATED = Path(OUTPUT_FOLDER) / "sentences_manually_curated.csv"

def load_data(csv_path):
    """Load and validate the dataset from CSV."""
    csv_path = Path(csv_path)
    logger.info(f"Reading input CSV from {csv_path}")
    df = pd.read_csv(csv_path, encoding=ENCODING, on_bad_lines="warn")
    
    # Add prompt_type
    gender_map = {"male": "M", "female": "F"}
    df["prompt_type"] = (
        df["adjective_gender"].map(gender_map) + 
        "→" + 
        df["noun_gender"].map(gender_map)
    )
    
    return df

def analyze_adjective_counts(df):
    """Analyze adjective counts in sentences."""
    # Calculate percentage of sentences with multiple adjectives
    pct_multi = (df['total_in_lists'] > 1).mean() * 100
    logger.info(f"Leakage summary — >1 count: {pct_multi:.2f}%")
    print(f"% sentences with more than one adjective: {pct_multi:.2f}%")
    
    # Calculate percentage of sentences with both male and female adjectives
    pct_co = ((df['male_count'] > 0) & (df['female_count'] > 0)).mean() * 100
    logger.info(f"Co-occurrence of male and female adjectives: {pct_co:.2f}%")
    print(f"% sentences with both male and female adjectives: {pct_co:.2f}%")
    
    return pct_multi, pct_co

def analyze_by_model_temperature(df):
    """Analyze leakage by model and temperature."""
    # Aggregate leakage by model × temperature
    agg_model_temp = (
        df
        .groupby(['model', 'temperature'], as_index=False)
        .apply(lambda sub: pd.Series({
            'pct_multi': (sub['total_in_lists'] > 1).mean() * 100
        }))
        .reset_index()
    )
    
    # Plot each model in its own color
    plt.figure(figsize=(8, 5))
    for model in agg_model_temp['model'].unique():
        subset = agg_model_temp[agg_model_temp['model'] == model]
        plt.plot(
            subset['temperature'],
            subset['pct_multi'],
            marker='o',
            label=model
        )
    plt.xlabel('Temperature')
    plt.ylabel('Leakage (%)')
    plt.title('Leakage vs. Temperature by Model')
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_FOLDER) / "leakage_by_model_temp.png")
    plt.close()
    
    return agg_model_temp

def analyze_by_prompt_structure(df):
    """Analyze leakage by prompt structure (gender combinations)."""
    # Calculate metrics by prompt type
    pt = df.groupby('prompt_type').apply(lambda sub: pd.Series({
        'pct_multi': (sub['total_in_lists'] > 1).mean() * 100,
        'pct_co': ((sub['male_count'] > 0) & (sub['female_count'] > 0)).mean() * 100
    })).reset_index()
    
    logger.info("Leakage by prompt structure:\n%s", pt.to_string())
    print("\nLeakage by prompt structure:\n", pt.to_string())
    
    # Create visualization
    labels = pt['prompt_type'].tolist()
    multi = pt['pct_multi'].values
    co = pt['pct_co'].values
    x = np.arange(len(labels))
    
    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.15, multi, width=0.3, label='Multi-adj')
    plt.bar(x + 0.15, co, width=0.3, label='Co-occur')
    plt.xticks(x, labels)
    plt.ylabel('Leakage (%)')
    plt.title('Leakage by Prompt Structure')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_FOLDER) / "leakage_by_prompt_structure.png")
    plt.close()
    
    return pt

def create_gender_stereotype_table(df):
    """Create a table showing gender stereotype consistency."""
    # Count each category
    count_MM = len(df[(df['noun_gender'] == 'male') & (df['adjective_gender'] == 'male')])
    count_FF = len(df[(df['noun_gender'] == 'female') & (df['adjective_gender'] == 'female')])
    count_MF = len(df[(df['noun_gender'] == 'male') & (df['adjective_gender'] == 'female')])
    count_FM = len(df[(df['noun_gender'] == 'female') & (df['adjective_gender'] == 'male')])
    
    # Totals
    total_S = count_MM + count_FF  # Stereotype consistent
    total_C = count_MF + count_FM  # Stereotype contradictory
    grand_total = len(df)
    
    # Percentages (relative to each half of the dataset)
    pct_MM = count_MM / total_S * 100
    pct_FF = count_FF / total_S * 100
    pct_MF = count_MF / total_C * 100
    pct_FM = count_FM / total_C * 100
    
    # Build summary table
    summary = pd.DataFrame({
        'Category': [
            'Consistent with gender stereotype',
            'Contradictory to gender stereotype',
            'Total'
        ],
        'Male Noun-Male Adj': [f"{count_MM} ({pct_MM:.1f}%)", f"{count_MF} ({pct_MF:.1f}%)", ''],
        'Female Noun-Female Adj': [f"{count_FF} ({pct_FF:.1f}%)", f"{count_FM} ({pct_FM:.1f}%)", ''],
        'Total': [
            f"{total_S} ({total_S/grand_total*100:.1f}%)",
            f"{total_C} ({total_C/grand_total*100:.1f}%)",
            f"{grand_total}"
        ]
    })
    
    logger.info("Gender stereotype table:\n%s", summary.to_string())
    print("\nTable: Labeling details with size & distribution\n", summary.to_string())
    
    # Save to CSV
    summary.to_csv(Path(OUTPUT_FOLDER) / "gender_stereotype_summary.csv", index=False)
    
    return summary

def analyze_adjective_distribution(df):
    """Analyze the distribution of adjectives per sentence."""
    # Count distribution of total_in_lists
    adj_counts = df['total_in_lists'].value_counts().sort_index()
    
    # Plot histogram of adjective counts
    plt.figure(figsize=(10, 6))
    adj_counts.plot(kind='bar')
    plt.xlabel('Number of Adjectives')
    plt.ylabel('Count of Sentences')
    plt.title('Distribution of Adjective Counts per Sentence')
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_FOLDER) / "adjective_count_distribution.png")
    plt.close()
    
    # Print statistics
    print("\nAdjective count distribution:")
    print(adj_counts)
    print(f"\nMean adjectives per sentence: {df['total_in_lists'].mean():.2f}")
    print(f"Median adjectives per sentence: {df['total_in_lists'].median()}")
    print(f"Max adjectives in a sentence: {df['total_in_lists'].max()}")
    
    return adj_counts

def manually_curate_sentences(df):
    """
    Manually curate sentences for analysis review.
    Presents sentences with unusual patterns for manual review and curation.
    """
    # Find sentences with unusual patterns to review
    # 1. Sentences with many adjectives (potential noise)
    many_adjs = df[df['total_in_lists'] > 3].copy()
    # 2. Sentences with both male and female adjectives (contradictions)
    both_genders = df[(df['male_count'] > 0) & (df['female_count'] > 0)].copy()
    
    # Combine sentences to review
    to_review = pd.concat([many_adjs, both_genders]).drop_duplicates()
    to_review.sort_values(by=['total_in_lists', 'male_count', 'female_count'], ascending=False, inplace=True)
    
    # Save to CSV for reference
    to_review.to_csv(CSV_REVIEW, index=False, encoding=ENCODING)
    logger.info(f"Saved {len(to_review)} sentences for review to {CSV_REVIEW}")
    
    # Interactive curation process
    manually_curated = []
    print(f"\n--- MANUAL CURATION OF ANALYSIS RESULTS ---")
    print(f"Total sentences to review: {len(to_review)}")
    
    for idx, row in to_review.iterrows():
        sentence = row['sentence']
        print(f"\n{idx}: {sentence}")
        print(f"  Male adjectives ({row['male_count']}): {row['male_matches']}")
        print(f"  Female adjectives ({row['female_count']}): {row['female_matches']}")
        print(f"  Total adjectives: {row['total_in_lists']}")
        
        decision = input("Keep this sentence in analysis? (y/n): ").strip().lower()
        if decision != 'y':
            manually_curated.append(idx)
    
    # Update the dataframe based on manual curation
    if manually_curated:
        # Create a new column to track manually curated status
        df['manually_excluded'] = False
        df.loc[manually_curated, 'manually_excluded'] = True
        
        # Create a new dataframe with only the included sentences
        curated_df = df[~df['manually_excluded']].copy()
        
        # Save the curated dataset
        curated_df.to_csv(CSV_CURATED, index=False, encoding=ENCODING)
        
        print(f"\nExcluded {len(manually_curated)} sentences through manual curation")
        print(f"Saved {len(curated_df)} curated sentences to {CSV_CURATED}")
        
        return curated_df
    else:
        print("\nNo sentences excluded during manual curation")
        return df

def main():
    """Main execution function."""
    try:
        # Load the data
        df = load_data(CSV_FINAL)
        
        # Print basic dataset information
        print(f"Dataset loaded with {len(df)} sentences")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Optional manual curation of sentences
        curation_choice = input("Do you want to manually curate sentences for analysis? (y/n): ").strip().lower()
        if curation_choice == 'y':
            df = manually_curate_sentences(df)
        
        # Analyze adjective counts
        pct_multi, pct_co = analyze_adjective_counts(df)
        
        # Analyze by model and temperature
        model_temp_stats = analyze_by_model_temperature(df)
        
        # Analyze by prompt structure
        prompt_structure_stats = analyze_by_prompt_structure(df)
        
        # Create gender stereotype table
        stereotype_table = create_gender_stereotype_table(df)
        
        # Analyze adjective distribution
        adj_distribution = analyze_adjective_distribution(df)
        
        # Print sample sentences for inspection
        print("\nSample sentences with both male & female adjectives:")
        both_genders = df[(df['male_count'] > 0) & (df['female_count'] > 0)].head(5)
        for _, row in both_genders.iterrows():
            print(f"- {row['sentence']}")
            print(f"  Male adjectives: {row['male_matches']}")
            print(f"  Female adjectives: {row['female_matches']}\n")
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())