import pandas as pd
import spacy
import logging

# Define numeric words to filter out
NUMERIC_WORDS = {
    "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen", "tien",
    "elf", "twaalf", "dertien", "veertien", "vijftien", "zestien", "zeventien",
    "achttien", "negentien", "twintig", "dertig", "veertig", "vijftig",
    "zestig", "zeventig", "tachtig", "negentig", "honderd", "duizend"
}

def contains_numeric_word(lemma):
    """Check if a lemma contains a numeric word."""
    return any(num in lemma for num in NUMERIC_WORDS)

def extract_adjectives_from_csv(file_path, nlp):
    """
    Reads a CSV file, parses each phrase with spaCy,
    and returns unique lemmatized adjectives.
    """
    logging.info(f"Loading CSV file: {file_path}")
    try:
        df = pd.read_csv(file_path, delimiter=';', usecols=[0], names=["Group"], header=0)
        logging.info(f"CSV loaded successfully with shape: {df.shape}")
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        raise
    
    df.dropna(subset=["Group"], inplace=True)
    logging.info(f"Dropped NaN rows. Remaining phrases: {len(df)}")
    
    adjectives = []
    logging.info("Starting POS tagging and lemmatization...")
    
    for idx, phrase in enumerate(df["Group"]):
        doc = nlp(phrase)
        for token in doc:
            if token.pos_ == "ADJ" and token.is_alpha:
                lemma = token.lemma_.lower()
                # Filter numbers except for 'een'
                if lemma != "een" and contains_numeric_word(lemma):
                    continue
                adjectives.append(lemma)
        
        if idx % 1000 == 0 and idx > 0:
            logging.info(f"Processed {idx} phrases...")
    
    unique_adjectives = list(dict.fromkeys(adjectives))
    logging.info(f"Extracted {len(unique_adjectives)} unique adjectives.")
    
    return unique_adjectives

def filter_adjectives(adjectives, model_w2v, model_ft, target_words):
    """Filter adjectives based on model vocabulary and target words."""
    adjectives_w2v = {w for w in adjectives if w in model_w2v}
    adjectives_ft = {w for w in adjectives if w in model_ft}
    
    # Take the intersection of the two sets
    union_vocab = adjectives_w2v.intersection(adjectives_ft)
    
    logging.info(f"Number of adjectives in Word2Vec: {len(adjectives_w2v)}")
    logging.info(f"Number of adjectives in FastText: {len(adjectives_ft)}")
    logging.info(f"Total in intersection (Word2Vec ∩ FastText): {len(union_vocab)}")
    
    # Exclude 'target_words' from the intersection
    filtered_adjectives = [
        adj for adj in union_vocab
        if not any(tw in adj for tw in target_words)
    ]
    
    logging.info(f"Remaining adjectives after filtering target words: {len(filtered_adjectives)}")
    
    return filtered_adjectives