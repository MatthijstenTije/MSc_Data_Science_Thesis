import logging
from pathlib import Path
import pandas as pd
import spacy
from collections import Counter
import re
from nltk.stem.snowball import DutchStemmer
from difflib import get_close_matches

from config import OUTPUT_FOLDER

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
CSV_FINAL = Path(OUTPUT_FOLDER) / "sentences_final.csv"
ENCODING = "utf-8"

# Define adjective sets
MALE_ADJS = {
    "corrupt", "onoverwinnelijk", "plaatsvervangend", "impopulair", "goddeloos", "incompetent",
    "misdadig", "bekwaam", "sadistisch", "gewetenloos", "steenrijk", "vooraanstaand", "voortvluchtig",
    "geniaal", "planmatig", "dood", "rebel", "islamistisch", "statutair", "schatrijk", "actief",
    "capabel", "overmoedig", "operationeel", "immoreel", "crimineel", "maffiose", "lucratief",
    "lamme", "onverbeterlijk"
}

FEMALE_ADJS = {
    "lesbisch", "blond", "beeldschoon", "ongepland", "bloedmooie", "beeldig", "sensueel", "platinablond",
    "voorlijk", "feministisch", "stijlvol", "tuttig", "huwelijks", "donkerharig", "ongehuwd", "kinderloos",
    "glamoureus", "rimpelig", "erotisch", "kleurig", "zilvergrijs", "rozig", "spichtig", "levenslustig",
    "hitsig", "rustiek", "teder", "marokkaans", "tenger", "exotisch"
}


def clean_punctuation(text: str) -> str:
    """Clean and normalize text."""
    import re
    import unicodedata
    # 1. Unicode normalization to split accents
    text = unicodedata.normalize("NFKD", text)
    # 2. Remove everything that's not a letter, number, or space
    text = re.sub(r"[^\w\s]", "", text)
    # 3. Reduce multiple spaces to single space
    text = re.sub(r"\s+", " ", text).strip()
    # 4. Convert to lowercase
    return text.lower()


def find_adjectives(text: str, male_adjs: set, female_adjs: set):
    """
    Find male and female adjectives in text using simple string matching.
    This approach doesn't rely on POS tagging or fuzzy matching, just exact word matching.
    
    Returns: male_count, female_count, male_hits, female_hits
    """
    # Convert to lowercase and split into words
    words = text.lower().split()
    
    # Find matches in the adjective sets
    male_hits = []
    female_hits = []
    
    for word in words:
        # Clean the word of any remaining punctuation
        clean_word = word.strip(".,;:!?\"'()[]{}")
        
        if clean_word in male_adjs:
            male_hits.append(clean_word)
        elif clean_word in female_adjs:
            female_hits.append(clean_word)
    
    # Deduplicate the lists
    male_hits = list(set(male_hits))
    female_hits = list(set(female_hits))
    
    return len(male_hits), len(female_hits), male_hits, female_hits


def adjective_matching(text: str, nlp, male_adjs: set, female_adjs: set):
    """
    Enhanced adjective matching that handles:
    1. Inflections & grammatical variants (comparative forms, gender agreements)
    2. Common spelling variations
    3. Stemming for root word matching
    4. Fuzzy matching for small variations
    
    Returns: male_count, female_count, male_hits, female_hits
    """
    # First try the existing simple matching for exact matches
    male_count, female_count, male_hits, female_hits = find_adjectives(text, male_adjs, female_adjs)
    
    # If we already found matches with simple string matching, we'll still continue
    # to find additional matches with advanced techniques
    
    # Initialize Dutch stemmer
    stemmer = DutchStemmer()
    
    # Get stems for all adjectives for matching
    male_stems = {stemmer.stem(adj) for adj in male_adjs}
    female_stems = {stemmer.stem(adj) for adj in female_adjs}
    
    # Process with spaCy
    doc = nlp(text)
    
    # Initialize sets to collect all unique matches
    all_male_hits = set(male_hits)
    all_female_hits = set(female_hits)
    
    # Track original words in text that led to matches
    original_words = set()
    
    for token in doc:
        word = token.text.lower().strip(".,;:!?\"'()[]{}")

        original_words.add(word)
            
        # Skip words we've already matched exactly
        if word in male_adjs or word in female_adjs:
            continue
            
        # Check if token is likely an adjective (based on POS tag)
        # Dutch adjective tags: ADJ, ADJX
        is_likely_adj = token.pos_ == "ADJ"
        
        # Try stemming to match root forms
        word_stem = stemmer.stem(word)
        
        # Match by stem
        if word_stem in male_stems:
            all_male_hits.add(word)
        elif word_stem in female_stems:
            all_female_hits.add(word)
        
        # For likely adjectives that haven't matched yet, try more techniques
        elif is_likely_adj:
            # Handle common Dutch adjectival suffixes
            base_forms = []
            
            # Try removing common Dutch adjective endings
            for suffix in ["e", "er", "ere", "ste", "ste", "er", "ste", "ste","oze","ere",""]:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    base_forms.append(word[:-len(suffix)])
            
            # Check if any base form matches our adjective lists
            for base in base_forms:
                if base in male_adjs or stemmer.stem(base) in male_stems:
                    all_male_hits.add(word)
                    break
                elif base in female_adjs or stemmer.stem(base) in female_stems:
                    all_female_hits.add(word)
                    break
        
    # Final fuzzy matching pass for remaining unmatched adjectives
    # Only consider words that look like adjectives but weren't matched yet
    remaining_words = original_words - all_male_hits - all_female_hits
    
    for word in remaining_words:
        # Only process words of reasonable length to avoid false positives
        if len(word) < 4:
            continue
            
        # Try fuzzy matching with relatively strict cutoff
        male_matches = get_close_matches(word, male_adjs, n=1, cutoff=0.85)
        if male_matches:
            all_male_hits.add(word)
            continue
            
        female_matches = get_close_matches(word, female_adjs, n=1, cutoff=0.85)
        if female_matches:
            all_female_hits.add(word)
    
    return len(all_male_hits), len(all_female_hits), list(all_male_hits), list(all_female_hits)


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
        .str.replace(r'\.+', '.', regex=True)  # Collapse multiple dots to one
        .str.replace(r'"', '', regex=True)
        # Don't remove all punctuation here to keep sentence structure
    )
    
    # Remove sentences that became empty after cleaning
    mask_cleaned_empty = df['sentence'].str.strip() == ''
    cleaned_empty_count = mask_cleaned_empty.sum()
    df = df[~mask_cleaned_empty]
    logger.info(f"Removed {cleaned_empty_count} empty sentences after final cleaning")
    
    logger.info(f"{len(df)} rows remain after cleaning")
    
    # Prepare spaCy model (disable parser for speed if not needed)
    logger.info("Loading spaCy model")
    nlp = spacy.load("nl_core_news_sm", disable=["ner"])
    
    # Count adjectives
    logger.info("Beginning sentence-level adjective counting with robust approach")
    
    # Create a clean version of sentences for adjective matching
    df['clean_sentence'] = df['sentence'].apply(clean_punctuation)
    
    # Apply the count_adjs function with our adjective sets
    results = df["clean_sentence"].apply(lambda text: adjective_matching(text, nlp, MALE_ADJS, FEMALE_ADJS))
    df[["male_count", "female_count", "male_matches", "female_matches"]] = \
        pd.DataFrame(results.tolist(), index=df.index)
    df["total_in_lists"] = df["male_count"] + df["female_count"]
    
    # Remove the temporary clean_sentence column
    df = df.drop('clean_sentence', axis=1)
    
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
    
    df.to_csv(CSV_OUT, index=False, encoding=ENCODING)
    logger.info(f"Saved cleaned & counted data to {CSV_OUT}")
    
    # Interactive curation of no-hits sentences
    no_hits_df = df[df["total_in_lists"] == 0]
    manually_curated = []

    print("\n--- MANUAL CURATION OF SENTENCES WITH NO ADJECTIVE MATCHES ---")
    print(f"Total sentences without matches: {len(no_hits_df)}")
    
    for idx, row in no_hits_df.iterrows():
        sentence = row['sentence']
        print(f"\n{idx}: {sentence}")
        decision = input("Include in final dataset? (y/n): ").strip().lower()
        if decision == 'y':
            manually_curated.append(idx)

    # Update the dataframes based on manual curation
    if manually_curated:
        # Mark these sentences as manually curated in the main dataframe
        df.loc[manually_curated, 'manually_curated'] = True
        
        # Update the final and no_hits dataframes
        final_df = df[(df["total_in_lists"] > 0) | (df['manually_curated'] == True)]
        no_hits_df = df[(df["total_in_lists"] == 0) & (df['manually_curated'] != True)]
        
        # Save updated dataframes
        final_df.to_csv(CSV_FINAL, index=False, encoding=ENCODING)
        no_hits_df[["sentence"]].to_csv(CSV_NO_HITS, index=False, encoding=ENCODING)
        
        print(f"\nAdded {len(manually_curated)} sentences to final dataset through manual curation")
    else:
        # Create no-hits file (original behavior)
        logger.info(f"Saving {len(no_hits_df)} sentences with no adjective hits to {CSV_NO_HITS}")
        no_hits_df[["sentence"]].to_csv(CSV_NO_HITS, index=False, encoding=ENCODING)
        
        # Create final file with only sentences that have adjective hits
        final_df = df[df["total_in_lists"] > 0]
        logger.info(f"Saving {len(final_df)} sentences with adjective hits to {CSV_FINAL}")
        final_df.to_csv(CSV_FINAL, index=False, encoding=ENCODING)
    
    return df


if __name__ == "__main__":
    main()