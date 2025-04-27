import os
import numpy as np
import pandas as pd
import logging
from scipy.stats import pearsonr
from config import FIGURES_DIR, TABLES_DIR

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def compute_bias(adj, male_terms, female_terms, model):
    """Compute gender bias for a single adjective."""
    male_mean = np.mean([cosine_similarity(model[adj], model[m]) for m in male_terms if m in model])
    female_mean = np.mean([cosine_similarity(model[adj], model[f]) for f in female_terms if f in model])
    return male_mean - female_mean

def compute_individual_bias(
    adjectives,
    male_terms,
    female_terms,
    model,
    exclude_substrings=True
):
    """
    Computes, for each adjective, the 'bias' difference between the average
    cosine similarity with male terms and the average cosine similarity
    with female terms.
    
    Parameters
    ----------
    adjectives : list of str
        List of adjectives to be analyzed (already cleaned/lemmatized).
    male_terms : list of str
        Words representing 'masculinity' (e.g., ['man', 'boy', 'father', ...]).
    female_terms : list of str
        Words representing 'femininity' (e.g., ['woman', 'girl', 'lady', ...]).
    model : dict-like of {str -> np.ndarray} or a KeyedVectors-like object
        Your embedding model, where you can check `word in model` and
        retrieve vectors using `model[word]`.
    exclude_substrings : bool, default=True
        Whether to exclude adjectives that contain any of the male or female
        terms as substrings (e.g., "manlike" contains "man").
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['word', 'male_mean', 'female_mean', 'bias_value'].
        Where 'bias_value' = male_mean - female_mean.
        Rows without sufficient embedding data are skipped.
    """
    # 1) filter out adjectives containing gender terms as substrings
    if exclude_substrings:
        all_target_words = set(male_terms + female_terms)
        def has_target_substring(adj):
            return any(tw in adj for tw in all_target_words)
        adjectives = [adj for adj in adjectives if not has_target_substring(adj)]
    
    records = []
    # 2) Loop over each adjective
    for adj in adjectives:
        if adj not in model:
            continue
        
        adj_vec = model[adj]
        
        # Gather cosine similarities with male terms
        male_sims = []
        for m in male_terms:
            if m in model:
                male_sims.append(cosine_similarity(adj_vec, model[m]))
        
        # Gather cosine similarities with female terms
        female_sims = []
        for f in female_terms:
            if f in model:
                female_sims.append(cosine_similarity(adj_vec, model[f]))
        
        # Skip if we can't compute both male and female means
        if len(male_sims) == 0 or len(female_sims) == 0:
            continue
        
        # Compute means
        male_mean = np.mean(male_sims)
        female_mean = np.mean(female_sims)
        
        # Compute bias
        bias_value = male_mean - female_mean
        
        records.append({
            "word": adj,
            "male_mean": male_mean,
            "female_mean": female_mean,
            "bias_value": bias_value
        })
    
    df_bias = pd.DataFrame(records)
    return df_bias

def compute_bias_with_pvalue(adj, male_terms, female_terms, model, permutations=1000, seed=42):
    """
    Computes the cosine-based gender bias of an adjective relative to male and female word sets,
    and estimates the statistical significance using a permutation test.
    
    Parameters
    ----------
    adj : str
        The adjective whose bias is being measured.
    male_terms : list of str
        A list of words representing the male group (e.g., ["man", "vader", "kerel"]).
    female_terms : list of str
        A list of words representing the female group (e.g., ["vrouw", "moeder", "meid"]).
    model : KeyedVectors or similar
        A word embedding model that supports word lookup and cosine operations (e.g., Word2Vec or FastText).
    permutations : int, default=1000
        The number of random shuffles to perform during the permutation test.
    seed : int, default=42
        Random seed to ensure reproducibility of the permutation results.
        
    Returns
    -------
    real_bias : float
        The observed bias score, defined as:
        mean_cosine(adj, male_terms) - mean_cosine(adj, female_terms)
    p_value : float
        The estimated probability that a bias of this magnitude (or stronger) could occur
        by chance if gender labels were random. Lower values indicate stronger significance.
    """
    # Ensure reproducibility
    np.random.seed(seed)
    
    # Step 1: Compute the real cosine-based bias score
    real_bias = compute_bias(adj, male_terms, female_terms, model)
    
    # Combine all gender terms into one pool to shuffle
    combined_terms = male_terms + female_terms
    num_male = len(male_terms)
    
    # Counter for how many permuted scores are as extreme as the real one
    extreme_count = 0
    
    # Step 2: Permutation loop
    for _ in range(permutations):
        np.random.shuffle(combined_terms)
        
        # Split shuffled terms into permuted male and female groups
        permuted_male = combined_terms[:num_male]
        permuted_female = combined_terms[num_male:]
        
        # Step 3: Compute bias under this random split
        permuted_bias = compute_bias(adj, permuted_male, permuted_female, model)
        
        # Count if the permuted bias is more extreme than the observed one
        if abs(permuted_bias) >= abs(real_bias):
            extreme_count += 1
    
    # Step 4: Compute the p-value as the proportion of extreme permutations
    p_value = extreme_count / permutations
    
    return real_bias, p_value

def tag_bias_agreement(row, alpha=0.05):
    """Tag rows based on bias agreement and significance."""
    sig_w2v = row['p_value_w2v'] < alpha
    sig_ft = row['p_value_ft'] < alpha
    same_sign = np.sign(row['bias_w2v']) == np.sign(row['bias_ft'])
    
    if sig_w2v and sig_ft:
        if same_sign:
            return "Significant in both (agree)"
        else:
            return "Significant in both (oppose)"
    elif sig_w2v:
        return "Only Word2Vec"
    elif sig_ft:
        return "Only FastText"
    else:
        return "Non-significant"

def save_fig(fname: str, **plt_kwargs):
    """Save the current figure into results/figures."""
    path = os.path.join(FIGURES_DIR, fname)
    import matplotlib.pyplot as plt
    plt.savefig(path, **plt_kwargs)

def save_table(df, fname: str, **to_csv_kwargs):
    """Save DataFrame to results/tables."""
    path = os.path.join(TABLES_DIR, fname)
    df.to_csv(path, **to_csv_kwargs)
