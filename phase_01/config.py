# Paths and constants for the analysis
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
FASTTEXT_MODEL_PATH = os.path.join(BASE_DIR, "../data", "cc.nl.300.vec.gz")
WORD2VEC_MODEL_PATH = os.path.join(BASE_DIR, "../data", "sonar-320.txt")
CSV_FILE_PATH = os.path.join(BASE_DIR, "../data", "Corpus_Hedendaags_Nederlands_Adjectives.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "output/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# sub‐folders
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR  = os.path.join(OUTPUT_DIR, "tables")

for d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR):
    os.makedirs(d, exist_ok=True)

# Gender words
MALE_WORDS = [
    "man", "kerel", "jongen", "vader", "zoon", "vent", "gast", 
    "meneer", "opa", "oom"
]

FEMALE_WORDS = [
    "vrouw", "dame", "meisje", "moeder", "dochter", "tante", "oma", 
    "mevrouw", "meid"
]


# Target words to exclude
TARGET_WORDS = [
    "man", "kerel", "vader", "zoon", "jongen", "vent", "gast", "meneer", "opa", "oom",
    "vrouw", "dame", "meisje", "moeder", "dochter", "tante", "oma", "mevrouw", "meid"
]