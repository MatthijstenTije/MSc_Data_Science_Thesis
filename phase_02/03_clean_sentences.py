# sentence_cleaner.py
import logging
from pathlib import Path
import pandas as pd
import spacy
from rapidfuzz import fuzz
from collections import Counter

from config import OUTPUT_FOLDER
from pathlib import Path
import logging
import pandas as pd
import spacy
from rapidfuzz import fuzz
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("sentence_cleaner")

# Configuration
CSV_IN = Path(OUTPUT_FOLDER) / "aggregated_sentences.csv"
CSV_OUT = Path(OUTPUT_FOLDER) / "sentences_cleaned.csv"
CSV_NO_HITS = Path(OUTPUT_FOLDER) / "sentences_no_hits.csv"
ENCODING = "utf-8"
FUZZY_THRESH = 70


def main():
    # Preflight & Load
    if not CSV_IN.exists():
        logger.error(f"Input file not found: {CSV_IN}")
        raise FileNotFoundError(f"{CSV_IN} does not exist")
    
    logger.info(f"Reading input CSV from {CSV_IN}")
    df = pd.read_csv(CSV_IN, encoding=ENCODING, on_bad_lines="warn")
    logger.info(f"Loaded {len(df)} rows")
    
    # Filter out rows with only dots, colons, or empty strings
    df['sentence'] = df['sentence'].fillna('').astype(str).str.strip()
    mask_dots = df['sentence'].str.match(r'^\.+\s*$', na=False)
    mask_colons = df['sentence'].str.match(r'^:+\s*$', na=False)
    mask_empty = df['sentence'].eq('')
    mask_removed = mask_dots | mask_colons | mask_empty
    removed_df = df[mask_removed]
    
    logger.info(f"Found {len(removed_df)} sentences to remove")
    
    # Log breakdown per category
    logger.info(
        "Removal breakdown – empty: %d, only dots: %d, only colons: %d",
        mask_empty.sum(),
        mask_dots.sum(),
        mask_colons.sum()
    )
    
    # Clean punctuation and normalize text
    df = df[~mask_removed].copy()
    df['sentence'] = (
        df['sentence']
        .str.strip()
        .str.replace(r'\.+', '', regex=True)
        .str.replace(r'"', '', regex=True)
        .apply(clean_punctuation)
    )
    
    # Remove sentences that became empty after cleaning
    mask_cleaned_empty = df['sentence'].str.strip() == ''
    cleaned_empty_count = mask_cleaned_empty.sum()
    df = df[~mask_cleaned_empty]
    logger.info(f"Removed {cleaned_empty_count} empty sentences after final cleaning")
    
    logger.info(f"{len(df)} rows remain after cleaning")
    
    # Prepare spaCy & adjective sets
    logger.info("Loading spaCy model")
    nlp = spacy.load("nl_core_news_sm", disable=["ner", "parser"])
    
    # Count adjectives
    logger.info("Beginning sentence-level adjective counting")
    results = df["sentence"].apply(lambda text: count_adjs(text, nlp))
    df[["male_count", "female_count", "male_matches", "female_matches"]] = \
        pd.DataFrame(results.tolist(), index=df.index)
    df["total_in_lists"] = df["male_count"] + df["female_count"]
    
    # Overall stats
    total_sentences = len(df)
    total_male_adjs = df["male_count"].sum()
    total_female_adjs = df["female_count"].sum()
    logger.info(
        "Processed %d sentences: total male adjectives=%d, total female adjectives=%d",
        total_sentences, total_male_adjs, total_female_adjs
    )
    
    # Top 5 adjectives by frequency
    male_flat = Counter(adj for lst in df["male_matches"] for adj in lst)
    female_flat = Counter(adj for lst in df["female_matches"] for adj in lst)
    
    logger.info(f"Top 5 male adjectives: {male_flat.most_common(5)}")
    logger.info(f"Top 5 female adjectives: {female_flat.most_common(5)}")
    
    # Save cleaned & counted data
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False, encoding=ENCODING)
    logger.info(f"Saved cleaned & counted data to {CSV_OUT}")
    
    # Create no-hits file
    CSV_NO_HITS = Path("Notebooks/Phase_02/output/sentences_no_hits.csv")
    CSV_NO_HITS.parent.mkdir(parents=True, exist_ok=True)
    no_hits_df = df[df["total_in_lists"] == 0]
    logger.info(f"Saving {len(no_hits_df)} sentences with no adjective hits to {CSV_NO_HITS}")
    no_hits_df[["sentence"]].to_csv(CSV_NO_HITS, index=False, encoding=ENCODING)
    
    return df

def clean_punctuation(s: str) -> str:
    """Clean and normalize text."""
    import re
    import unicodedata
    # 1. Unicode-normaliseren zodat accenten gesplitst worden
    s = unicodedata.normalize("NFKD", s)
    # 2. alles wat geen letter, cijfer of spatie is, weghalen
    s = re.sub(r"[^\w\s]", "", s)
    # 3. meerdere spaties terugbrengen tot één
    s = re.sub(r"\s+", " ", s).strip()
    # 4. naar lowercase
    return s.lower()

def count_adjs(text: str, nlp):
    """
    Returns: male_count, female_count, male_hits, female_hits
    """
    # Define adjective sets
    male_adjs = {
        "corrupt", "incompetent", "onoverwinnelijk", "misdadig",
        "plaatsvervangend", "bekwaam", "sadistisch", "impopulair",
        "gewetenloos", "goddeloos", "steenrijk", "vooraanstaand",
        "voortvluchtig", "geniaal", "planmatig", "bekwaamheid",
        "genialiteit"
    }
    female_adjs = {
        "blond", "beeldschoon", "bloedmooie", "kinderloos",
        "voorlijk", "glamoureus", "feministisch", "beeldig",
        "stijlvol", "donkerharig", "sensueel", "tuttig",
        "ongehuwd", "platinablond", "rimpelig"
    }
    
    doc = nlp(text)
    male_hits = []
    female_hits = []
    
    for tok in doc:
        lemma = tok.lemma_.lower()
        # 1) Exact lemma match in male/female lijsten
        if lemma in male_adjs:
            male_hits.append(lemma)
        elif lemma in female_adjs:
            female_hits.append(lemma)
        else:
            # 2) Fuzzy fallback op lemma
            best_male = max(male_adjs, key=lambda a: fuzz.ratio(lemma, a))
            best_fem = max(female_adjs, key=lambda a: fuzz.ratio(lemma, a))
            score_m = fuzz.ratio(lemma, best_male)
            score_f = fuzz.ratio(lemma, best_fem)
            if score_m >= FUZZY_THRESH and score_m >= score_f:
                male_hits.append(best_male)
            elif score_f >= FUZZY_THRESH:
                female_hits.append(best_fem)
    
    # per zin dedupliceren
    return len(set(male_hits)), len(set(female_hits)), list(set(male_hits)), list(set(female_hits))

if __name__ == "__main__":
    main()