import logging
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress
from sklearn.linear_model import LinearRegression
from wefe.query import Query
from wefe.metrics.RIPA import RIPA
import logging
from gensim.models import KeyedVectors
from wefe.word_embedding_model import WordEmbeddingModel
from models import load_word2vec_model, load_fasttext_model
from config import TABLES_DIR, FASTTEXT_MODEL_PATH, WORD2VEC_MODEL_PATH, BASE_DIR

fasttext_model = load_fasttext_model(FASTTEXT_MODEL_PATH)
model_w2v = load_word2vec_model(WORD2VEC_MODEL_PATH)

class GensimDutchEmbeddingModel(WordEmbeddingModel):
    def __init__(self, keyed_vectors):
        super().__init__(wv=keyed_vectors)

def run_ripa_analysis(filtered_adjectives, male_words, female_words, model_w2v, model_ft, df_bias_sig_w2v, df_bias_sig_ft):
    """Run RIPA analysis on word embeddings."""
    logging.info("Starting RIPA analysis...")
    
    # Create WEFE-compatible models
    w2v_model = GensimDutchEmbeddingModel(model_w2v)
    fasttext_model = model_ft 
    
    # Define the query
    query = Query(
        target_sets=[
        ["man", "kerel", "jongen", "vader", "zoon", "vent", "meneer", "opa", "oom"],
        ["vrouw", "dame", "meisje", "moeder", "dochter", "tante", "oma", "mevrouw", "meid"]
        ],
        attribute_sets=[filtered_adjectives],
        target_sets_names=["Male Terms", "Female Terms"],
        attribute_sets_names=["Adjectives"],
    )
    
    # Run RIPA
    ripa = RIPA()
    logging.info("Running RIPA on Word2Vec model...")
    result_ripa_w2v = ripa.run_query(query, w2v_model)
    logging.info("Running RIPA on FastText model...")
    result_ripa_ft = ripa.run_query(query, fasttext_model)
    
    # Process results
    df_ripa_w2v = pd.DataFrame({
        'Word': result_ripa_w2v["word_values"].keys(),
        'Mean Score': [val['mean'] for val in result_ripa_w2v["word_values"].values()],
        'Std Dev': [val['std'] for val in result_ripa_w2v["word_values"].values()],
    })
    
    df_ripa_ft = pd.DataFrame({
        'Word': result_ripa_ft["word_values"].keys(),
        'Mean Score': [val['mean'] for val in result_ripa_ft["word_values"].values()],
        'Std Dev': [val['std'] for val in result_ripa_ft["word_values"].values()],
    })
    
    # Sort on Mean Score
    df_ripa_w2v = df_ripa_w2v.sort_values(by="Mean Score", ascending=False).reset_index(drop=True)
    df_ripa_ft = df_ripa_ft.sort_values(by="Mean Score", ascending=False).reset_index(drop=True)
    
    # Calculate Z-scores
    mean_of_scores_w2v = df_ripa_w2v["Mean Score"].mean()
    std_of_scores_w2v = df_ripa_w2v["Mean Score"].std()
    df_ripa_w2v["Z-Score"] = (df_ripa_w2v["Mean Score"] - mean_of_scores_w2v) / std_of_scores_w2v
    df_ripa_w2v = df_ripa_w2v.sort_values("Z-Score", ascending=False).reset_index(drop=True)
    
    mean_of_scores_ft = df_ripa_ft["Mean Score"].mean()
    std_of_scores_ft = df_ripa_ft["Mean Score"].std()
    df_ripa_ft["Z-Score"] = (df_ripa_ft["Mean Score"] - mean_of_scores_ft) / std_of_scores_ft
    df_ripa_ft = df_ripa_ft.sort_values("Z-Score", ascending=False).reset_index(drop=True)
    
    # Combine with previous results
    df_combined_w2v = prepare_bias_comparison(df_bias_sig_w2v, df_ripa_w2v)
    df_combined_ft = prepare_bias_comparison(df_bias_sig_ft, df_ripa_ft)
    
    # Calculate correlations
    r_w2v, p_w2v = pearsonr(df_combined_w2v['bias'], df_combined_w2v['RIPA_score'])
    r_ft, p_ft = pearsonr(df_combined_ft['bias'], df_combined_ft['RIPA_score'])
    
    logging.info(f"Word2Vec: Correlation between cosine bias and RIPA: r = {r_w2v:.3f}, p = {p_w2v:.4f}")
    logging.info(f"FastText: Correlation between cosine bias and RIPA: r = {r_ft:.3f}, p = {p_ft:.4f}")
    
    # Calculate linear regression R² scores
    X_w2v = df_combined_w2v[['bias']].values
    X_ft = df_combined_ft[['bias']].values
    y_w2v = df_combined_w2v['RIPA_score'].values
    y_ft = df_combined_ft['RIPA_score'].values
    
    model_w2v_reg = LinearRegression().fit(X_w2v, y_w2v)
    model_ft_reg = LinearRegression().fit(X_ft, y_ft)
    
    r2_w2v = model_w2v_reg.score(X_w2v, y_w2v)
    r2_ft = model_ft_reg.score(X_ft, y_ft)
    
    logging.info(f"Word2Vec R²: {r2_w2v:.3f}")
    logging.info(f"FastText R²: {r2_ft:.3f}")
    
    # Calculate word length correlations
    df_combined_w2v['adjective_length'] = df_combined_w2v['word'].str.len()
    df_combined_ft['adjective_length'] = df_combined_ft['word'].str.len()
    
    length_slope_w2v, _, length_r_w2v, length_p_w2v, _ = linregress(df_combined_w2v['adjective_length'], df_combined_w2v['ripa_z'])
    length_slope_ft, _, length_r_ft, length_p_ft, _ = linregress(df_combined_ft['adjective_length'], df_combined_ft['ripa_z'])
    
    logging.info(f"Word2Vec: Length vs RIPA Z-score: r = {length_r_w2v:.3f}, p = {length_p_w2v:.4f}, slope = {length_slope_w2v:.3f}")
    logging.info(f"FastText: Length vs RIPA Z-score: r = {length_r_ft:.3f}, p = {length_p_ft:.4f}, slope = {length_slope_ft:.3f}")
    
    # Calculate consistency and divergence
    df_combined_w2v['abs_diff'] = abs(df_combined_w2v['cosine_bias_z'] - df_combined_w2v['ripa_z'])
    df_combined_ft['abs_diff'] = abs(df_combined_ft['cosine_bias_z'] - df_combined_ft['ripa_z'])
    
    logging.info("\n--- Top 5 Most Consistent Words (W2V) ---")
    for _, row in df_combined_w2v.sort_values('abs_diff').head(5).iterrows():
        logging.info(f"{row['word']}: cosine_z = {row['cosine_bias_z']:.3f}, ripa_z = {row['ripa_z']:.3f}")
    
    logging.info("\n--- Top 5 Most Divergent Words (W2V) ---")
    for _, row in df_combined_w2v.sort_values('abs_diff', ascending=False).head(5).iterrows():
        logging.info(f"{row['word']}: cosine_z = {row['cosine_bias_z']:.3f}, ripa_z = {row['ripa_z']:.3f}")
    
    # Get top gender-biased words
    male_top_w2v, female_top_w2v = get_top_gender_biased_words(df_combined_w2v, top_n=35, method='combined')
    
    logging.info(f"Saving top male and female biased adjectives to CSV files...")
    male_top_w2v['word'].to_csv(
    os.path.join(TABLES_DIR, "top_male_biased_adjectives_w2v.csv"),
    index=False, header=False
    )
    female_top_w2v['word'].to_csv(
        os.path.join(TABLES_DIR, "top_female_biased_adjectives_w2v.csv"),
        index=False, header=False
        )
    
    return df_ripa_w2v, df_ripa_ft, df_combined_w2v, df_combined_ft, male_top_w2v, female_top_w2v

def prepare_bias_comparison(df_cosine, df_ripa):
    """
    Merge cosine similarity bias with RIPA scores, compute Z-scores, and return full merged DataFrame.
    """
    df = pd.merge(
        df_cosine,
        df_ripa[['Word', 'Mean Score']],
        left_on='word',
        right_on='Word',
        how='inner'
    ).rename(columns={'Mean Score': 'RIPA_score'})
    
    # Z-score normalize both metrics
    df['cosine_bias_z'] = (df['bias'] - df['bias'].mean()) / df['bias'].std()
    df['ripa_z'] = (df['RIPA_score'] - df['RIPA_score'].mean()) / df['RIPA_score'].std()
    
    return df.drop(columns='Word')

def get_top_gender_biased_words(df, top_n=35, method='combined'):
    """
    Returns top N male- and female-biased adjectives based on both cosine and RIPA z-scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['word', 'cosine_bias_z', 'ripa_z']
    top_n : int
        Number of top biased adjectives to return for each gender
    method : str
        Scoring method: 'combined', 'ripa', 'cosine', or 'borda'
        
    Returns
    -------
    df_male_top : Top N male-biased adjectives
    df_female_top : Top N female-biased adjectives
    """
    df = df.copy()
    
    if method == 'combined':
        df['score'] = (df['cosine_bias_z'] + df['ripa_z']) / 2
    elif method == 'borda':
        df['rank_cosine'] = df['cosine_bias_z'].rank(ascending=False)
        df['rank_ripa'] = df['ripa_z'].rank(ascending=False)
        df['score'] = df['rank_cosine'] + df['rank_ripa']
    elif method == 'ripa':
        df['score'] = df['ripa_z']
    elif method == 'cosine':
        df['score'] = df['cosine_bias_z']
    else:
        raise ValueError("Method must be 'combined', 'borda', 'ripa', or 'cosine'")
    
    # Sort by final score (desc → male bias; asc → female bias)
    df_male = df.sort_values('score', ascending=False).head(top_n)
    df_female = df.sort_values('score', ascending=True).head(top_n)
    
    return df_male[['word', 'cosine_bias_z', 'ripa_z', 'score']], df_female[['word', 'cosine_bias_z', 'ripa_z', 'score']]