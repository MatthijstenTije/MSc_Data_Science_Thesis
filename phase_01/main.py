import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
    )
import spacy
from config import (
    FASTTEXT_MODEL_PATH, WORD2VEC_MODEL_PATH, CSV_FILE_PATH,
    MALE_WORDS, FEMALE_WORDS, TARGET_WORDS
)
from models import load_fasttext_model, load_word2vec_model
from preprocessing import extract_adjectives_from_csv, filter_adjectives
from bias_metrics import (
    compute_raw_biases, compute_permutation_tests, 
    merge_bias_results, compute_z_scores,
)

from visualization import (
    plot_bias_comparison, plot_bias_barplot_abs, 
    plot_bias_correlation, plot_ripa_vs_length_with_stats,
    plot_bias_venn_diagram
)
from ripa_analysis import run_ripa_analysis

def main():
    
    # 1. Load spaCy for Dutch language processing
    logging.info("Loading spaCy Dutch language model...")
    nlp = spacy.load('nl_core_news_lg')
    
    # 2. Load embedding models
    fasttext_model = load_fasttext_model(FASTTEXT_MODEL_PATH)
    model_w2v = load_word2vec_model(WORD2VEC_MODEL_PATH)
    
    # 3. Extract adjectives from CSV
    adjectives = extract_adjectives_from_csv(CSV_FILE_PATH, nlp)
    
    # 4. Filter adjectives based on model vocabulary and target words
    filtered_adjectives = filter_adjectives(adjectives, model_w2v, fasttext_model, TARGET_WORDS)
    
    # 5. Compute raw biases
    df_indiv_bias_w2v, df_indiv_bias_ft, top_bias_words_w2v, top_bias_words_ft = compute_raw_biases(
        filtered_adjectives, MALE_WORDS, FEMALE_WORDS, model_w2v, fasttext_model
    )
    
    # 6. Compute permutation tests
    df_bias_sig_w2v, df_bias_sig_ft = compute_permutation_tests(
        filtered_adjectives, MALE_WORDS, FEMALE_WORDS, model_w2v, fasttext_model
    )
    
    # 7. Merge bias results
    df_compare, corr, pval = merge_bias_results(df_bias_sig_w2v, df_bias_sig_ft)
    
    # 8. Compute Z-scores and filter by gender bias direction
    df_z_compare, df_male, df_female = compute_z_scores(df_bias_sig_w2v, df_bias_sig_ft)
    
    # 9. Create visualizations
    logging.info("Creating bias comparison scatter plot...")
    plot_bias_comparison(df_compare)
    
    logging.info("Creating male-biased adjectives bar plot...")
    plot_bias_barplot_abs(df_male, "Top Male-Biased Adjectives (Absolute Z-Score Comparison)")
    
    logging.info("Creating female-biased adjectives bar plot...")
    plot_bias_barplot_abs(df_female, "Top Female-Biased Adjectives (Absolute Z-Score Comparison)")
    
    # 10. Run RIPA analysis
    logging.info("Running RIPA analysis...")
    df_ripa_w2v, df_ripa_ft, df_combined_w2v, df_combined_ft, male_top_w2v, female_top_w2v = run_ripa_analysis(
        filtered_adjectives, MALE_WORDS, FEMALE_WORDS, model_w2v, fasttext_model, df_bias_sig_w2v, df_bias_sig_ft
    )
    
    # 11. Create RIPA-related visualizations
    logging.info("Creating RIPA correlation plots...")
    plot_bias_correlation(df_combined_w2v, "Word2Vec")
    plot_bias_correlation(df_combined_ft, "FastText")
    
    logging.info("Creating RIPA vs adjective length plots...")
    plot_ripa_vs_length_with_stats(df_combined_w2v, "Word2Vec", "#1f77b4")
    plot_ripa_vs_length_with_stats(df_combined_ft, "FastText", "#ff7f0e")
     
    # 12. Create Venn Diagram
    logging.info("Creating Venn Diagram")
    plot_bias_venn_diagram(
        df_combined_w2v,
        df_combined_ft,
        df_combined_w2v,
        250
    )
    
    logging.info("Analysis complete!")

if __name__ == "__main__":
    main()