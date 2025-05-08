import pandas as pd
import re
import os
import spacy
from models import load_word2vec_model
from config import WORD2VEC_MODEL_PATH

def get_clean_adjectives_from_tsv(file_path, embedding_model, nlp_model):
    """
    Extracts and filters Dutch adjectives from a TSV file using spaCy and an embedding model.
    """

    def clean_lemma(lemma):
        lemma = lemma.lower()
        match = re.match(r"[a-z]+", lemma)
        return match.group(0) if match else ''

    # Load and filter the TSV data
    df = pd.read_csv(file_path, delimiter='\t', header=None)
    df = df[df[5] == "AA(degree=pos,case=norm,infl=e)"]

    df['cleaned_lemma'] = df[1].apply(clean_lemma)
    lemma_list = [lemma for lemma in df['cleaned_lemma'] if lemma and len(lemma) > 1]

    unique_lemmas = list(dict.fromkeys(lemma_list))
    print(f"Total unique lemmas: {len(unique_lemmas)}")

    missing = [word for word in unique_lemmas if word not in embedding_model]
    print(f"Missing in embedding model: {len(missing)} (e.g., {missing[:10]})")

    present_lemmas = [word for word in unique_lemmas if word in embedding_model]

    def filter_adjectives_spacy(lemmas, nlp):
        return [lemma for lemma in lemmas if nlp(lemma)[0].pos_ == 'ADJ']

    filtered_adjectives = filter_adjectives_spacy(present_lemmas, nlp_model)

    target_words = [
        'man', 'mannen', 'jongen', 'kerel', 'vader', 'zoon',
        'vrouw', 'vrouwelijk', 'vrouwen', 'meisje', 'dame'
    ]

    final_adjectives = [
        adj for adj in filtered_adjectives
        if not any(t in adj for t in target_words)
    ]

    print(f"Final adjectives after filtering: {len(final_adjectives)}")
    return final_adjectives


print("Loading spaCy and Word2Vec models...")
nlp = spacy.load("nl_core_news_lg")
model = load_word2vec_model(WORD2VEC_MODEL_PATH)

path = os.path.join("data", "molex_22_02_2022.tsv")

# Run extraction
adjectives = get_clean_adjectives_from_tsv(path, model, nlp)