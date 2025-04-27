import logging
import numpy as np
import pandas as pd
from wefe.query import Query
from wefe.metrics.RIPA import RIPA
from utils import compute_individual_bias, compute_bias_with_pvalue, tag_bias_agreement
from scipy.stats import pearsonr

def compute_raw_biases(filtered_adjectives, male_words, female_words, model_w2v, model_ft):
    """Compute raw bias scores for adjectives using both models."""
    # STEP 1 — WORD2VEC
    logging.info("Step 1 (W2V): Computing raw cosine biases for all adjectives...")
    df_indiv_bias_w2v = compute_individual_bias(
        adjectives=filtered_adjectives,
        male_terms=male_words,
        female_terms=female_words,
        model=model_w2v,
        exclude_substrings=True  # Optional, avoids words like "mannetje"
    )
    logging.info(f"Word2Vec: Computed raw bias for {len(df_indiv_bias_w2v)} adjectives.")
    
    # STEP 1 — FASTTEXT
    logging.info("Step 1 (FastText): Computing raw cosine biases for all adjectives...")
    df_indiv_bias_ft = compute_individual_bias(
        adjectives=filtered_adjectives,
        male_terms=male_words,
        female_terms=female_words,
        model=model_ft,
        exclude_substrings=True
    )
    logging.info(f"FastText: Computed raw bias for {len(df_indiv_bias_ft)} adjectives.")
    
    # STEP 2 — Calculate absolute bias and get top biased words
    df_indiv_bias_w2v['abs_bias'] = df_indiv_bias_w2v['bias_value'].abs()
    top_bias_words_w2v = df_indiv_bias_w2v.sort_values('abs_bias', ascending=False)['word'].tolist()
    
    df_indiv_bias_ft['abs_bias'] = df_indiv_bias_ft['bias_value'].abs()
    top_bias_words_ft = df_indiv_bias_ft.sort_values('abs_bias', ascending=False)['word'].tolist()
    
    logging.info(f"Most biased words (W2V):")
    logging.info(str(top_bias_words_w2v[:10]))
    logging.info(f"Most biased words (FastText):")
    logging.info(str(top_bias_words_ft[:10]))
    
    return df_indiv_bias_w2v, df_indiv_bias_ft, top_bias_words_w2v, top_bias_words_ft

def compute_permutation_tests(filtered_adjectives, male_words, female_words, model_w2v, model_ft):
    """Run permutation tests for statistical significance."""
    # STEP 3 — WORD2VEC
    results_w2v = []
    logging.info(f"Step 3 (W2V): Running permutation p-value test on adjectives...")
    for word in filtered_adjectives:
        try:
            bias, pval = compute_bias_with_pvalue(word, male_words, female_words, model_w2v)
            results_w2v.append({'word': word, 'bias': bias, 'p_value': pval})
        except Exception as e:
            logging.warning(f"Word2Vec error on '{word}': {e}")
    
    df_bias_sig_w2v = pd.DataFrame(results_w2v).sort_values('p_value')
    logging.info(f"Word2Vec: Finished permutation testing for {len(results_w2v)} words.")
    logging.info("\n=== Top Word2Vec Results (sorted by p-value) ===")
    logging.info(str(df_bias_sig_w2v.head(10)))
    
    # STEP 3 — FASTTEXT
    logging.info(f"Step 3 (FastText): Running permutation p-value test on adjectives...")
    results_ft = []
    for word in filtered_adjectives:
        try:
            bias, pval = compute_bias_with_pvalue(word, male_words, female_words, model_ft)
            results_ft.append({'word': word, 'bias': bias, 'p_value': pval})
        except Exception as e:
            logging.warning(f"FastText error on '{word}': {e}")
    
    df_bias_sig_ft = pd.DataFrame(results_ft).sort_values('p_value')
    logging.info(f"FastText: Finished permutation testing for {len(results_ft)} words.")
    logging.info("\n=== Top FastText Results (sorted by p-value) ===")
    logging.info(str(df_bias_sig_ft.head(10)))
    
    return df_bias_sig_w2v, df_bias_sig_ft

def merge_bias_results(df_bias_sig_w2v, df_bias_sig_ft):
    """Merge bias results from Word2Vec and FastText for comparison."""
    # Merge on word
    df_compare = pd.merge(
        df_bias_sig_w2v.rename(columns={'bias': 'bias_w2v', 'p_value': 'p_value_w2v'}),
        df_bias_sig_ft.rename(columns={'bias': 'bias_ft', 'p_value': 'p_value_ft'}),
        on='word',
        suffixes=('', '_ft'),
    )
    
    # Tag each row
    df_compare['tag'] = df_compare.apply(tag_bias_agreement, axis=1)
    
    # Calculate correlation
    corr, pval = pearsonr(df_compare['bias_w2v'], df_compare['bias_ft'])
    logging.info(f"Correlation (Word2Vec vs. FastText) = {corr:.3f} (p = {pval:.4g})")
    
    return df_compare, corr, pval

def compute_z_scores(df_bias_sig_w2v, df_bias_sig_ft):
    """Compute Z-scores for bias values."""
    # Z-scores for Word2Vec
    mean_w2v = df_bias_sig_w2v['bias'].mean()
    std_w2v = df_bias_sig_w2v['bias'].std()
    df_bias_sig_w2v['z_score_w2v'] = (df_bias_sig_w2v['bias'] - mean_w2v) / std_w2v
    
    # Z-scores for FastText
    mean_ft = df_bias_sig_ft['bias'].mean()
    std_ft = df_bias_sig_ft['bias'].std()
    df_bias_sig_ft['z_score_ft'] = (df_bias_sig_ft['bias'] - mean_ft) / std_ft
    
    # Merge on shared adjectives
    df_z_compare = pd.merge(
        df_bias_sig_w2v[['word', 'z_score_w2v']],
        df_bias_sig_ft[['word', 'z_score_ft']],
        on='word'
    )
    
    # Select top N based on absolute average Z-score
    df_z_compare['avg_abs_z'] = (df_z_compare['z_score_w2v'].abs() + df_z_compare['z_score_ft'].abs()) / 2
    df_top = df_z_compare.sort_values('avg_abs_z', ascending=False).head(30)
    
    logging.info("\n=== Top 30 Biased Words Across Both Models (by average Z) ===")
    logging.info(str(df_top[['word', 'z_score_w2v', 'z_score_ft']]))
    
    # Filter by gender bias direction
    df_male = df_z_compare[
        (df_z_compare['z_score_w2v'] > 0) & (df_z_compare['z_score_ft'] > 0)
    ].copy()
    df_female = df_z_compare[
        (df_z_compare['z_score_w2v'] < 0) & (df_z_compare['z_score_ft'] < 0)
    ].copy()
    
    # Top-N limit
    TOP_N = 10
    df_male = df_male.sort_values('avg_abs_z', ascending=False).head(TOP_N)
    df_female = df_female.sort_values('avg_abs_z', ascending=False).head(TOP_N)
    
    return df_z_compare, df_male, df_female