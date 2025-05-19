import pandas as pd
from pathlib import Path
from ast import literal_eval
from config import OUTPUT_FOLDER

# === CONFIG ===
INPUT_FILE = "sentences_final.csv"
OUTPUT_FILE = "sentence_level_odds_ratios_by_temperature.csv"
DATA_DIR = Path(OUTPUT_FOLDER)

# === LOAD DATA ===
df = pd.read_csv(DATA_DIR / INPUT_FILE)
df.columns = df.columns.str.strip().str.lower()
df['word'] = df['word'].astype(str).str.lower().str.strip()
df['noun_gender'] = df['noun_gender'].astype(str).str.lower().str.strip()
df = df[df['noun_gender'].isin(['male', 'female'])]

# === Detection based on extra adjectives  ===
def has_extra(row, match_col):
    try:
        base = row.get("word", "").strip().lower()
        matches = literal_eval(row[match_col]) if pd.notna(row[match_col]) else []
        matches = [w.strip().lower() for w in matches if isinstance(w, str)]
        return any(w != base for w in matches)
    except:
        return False

df['has_extra_male'] = df.apply(lambda r: has_extra(r, 'male_matches'), axis=1)
df['has_extra_female'] = df.apply(lambda r: has_extra(r, 'female_matches'), axis=1)

# === Odds ratio per temperature ===
epsilon = 1e-6
results = []

for temperature, group in df.groupby('temperature'):
    Dm = len(group[group['noun_gender'] == 'male'])
    Df = len(group[group['noun_gender'] == 'female'])

    Dm_m = len(group[(group['noun_gender'] == 'male') & (group['has_extra_male'])])
    Df_m = len(group[(group['noun_gender'] == 'female') & (group['has_extra_male'])])
    Df_f = len(group[(group['noun_gender'] == 'female') & (group['has_extra_female'])])
    Dm_f = len(group[(group['noun_gender'] == 'male') & (group['has_extra_female'])])

    Dm_not_m = Dm - Dm_m
    Df_not_m = Df - Df_m
    Df_not_f = Df - Df_f
    Dm_not_f = Dm - Dm_f

    ORm = ((Dm_m + epsilon) / (Dm_not_m + epsilon)) / ((Df_m + epsilon) / (Df_not_m + epsilon))
    ORf = ((Df_f + epsilon) / (Df_not_f + epsilon)) / ((Dm_f + epsilon) / (Dm_not_f + epsilon))

    results.append({
        "temperature": temperature,
        "total_sentences_male": Dm,
        "total_sentences_female": Df,
        "Dm_m": Dm_m,
        "Df_m": Df_m,
        "Df_f": Df_f,
        "Dm_f": Dm_f,
        "ORm": ORm,
        "ORf": ORf,
        "|ORm - 1| + |ORf - 1|": abs(ORm - 1) + abs(ORf - 1),
    })

# === Save and display ===
or_df = pd.DataFrame(results)
or_df['bias_rank'] = or_df['|ORm - 1| + |ORf - 1|'].rank(method='min')
or_df = or_df.sort_values('bias_rank')

or_df.to_csv(DATA_DIR / OUTPUT_FILE, index=False)
print(f"[✓] Saved OR results by temperature to: {OUTPUT_FILE}")
print(or_df[['temperature', 'ORm', 'ORf', '|ORm - 1| + |ORf - 1|', 'bias_rank']])
