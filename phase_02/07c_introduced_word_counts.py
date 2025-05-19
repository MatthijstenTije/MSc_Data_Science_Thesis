import logging
from pathlib import Path
import pandas as pd
from config import OUTPUT_FOLDER

input_path = Path(OUTPUT_FOLDER) / "sentences_final.csv"

# Load dataset
df = pd.read_csv(input_path)

# Step 1: words that appear in the 'word' column
original_words = set(df['word'].unique())

# Step 2: function to extract words from column (from list string)
def extract_words(series):
    words = set()
    for row in series.dropna():
        row = row.strip("[]").replace("'", "").replace('"', '')
        words.update(w.strip() for w in row.split(",") if w.strip())
    return words

# Step 3: all words that appear in male_matches and female_matches
matched_words = extract_words(df['male_matches']) | extract_words(df['female_matches'])

# Step 4: extra words that appear in matches but not in 'word'
extra_matched = matched_words - original_words

# Function to count extra words per row
def count_extras(row, col):
    if pd.isna(row[col]):
        return 0
    words = [w.strip() for w in row[col].strip("[]").replace("'", "").replace('"', '').split(",")]
    return sum(1 for w in words if w in extra_matched)

df['extra_male'] = df.apply(lambda row: count_extras(row, 'male_matches'), axis=1)
df['extra_female'] = df.apply(lambda row: count_extras(row, 'female_matches'), axis=1)
df['extra_total'] = df['extra_male'] + df['extra_female']

# Step 5: group by gender alignment
df['gender_match'] = df['noun_gender'] == df['adjective_gender']

# Result: totals per type of comparison
summary = df.groupby('gender_match')['extra_total'].sum().reset_index()
summary['gender_relation'] = summary['gender_match'].map({
    True: "noun_gender = adjective_gender",
    False: "noun_gender ≠ adjective_gender"
})
summary = summary[['gender_relation', 'extra_total']]

# Print or save
print(summary)

# Save if desired:
summary.to_csv(Path(OUTPUT_FOLDER) / "extra_matches_by_gender_relation.csv", index=False)

from collections import Counter

# Function to count all matches per column, but only for extra words
def count_word_frequencies(series, valid_words):
    freq = Counter()
    for row in series.dropna():
        row = row.strip("[]").replace("'", "").replace('"', '')
        words = [w.strip() for w in row.split(",") if w.strip()]
        for w in words:
            if w in valid_words:
                freq[w] += 1
    return freq

# Frequency per extra word in male/female matches
male_freq = count_word_frequencies(df['male_matches'], extra_matched)
female_freq = count_word_frequencies(df['female_matches'], extra_matched)

# Combine and create dataframe
combined = pd.DataFrame([
    {
        'word': word,
        'male_count': male_freq.get(word, 0),
        'female_count': female_freq.get(word, 0),
        'total': male_freq.get(word, 0) + female_freq.get(word, 0)
    }
    for word in sorted(extra_matched)
])
detailed_rows = []

for _, row in df.iterrows():
    for match_col in ['male_matches', 'female_matches']:
        if pd.isna(row[match_col]):
            continue
        matches = row[match_col].strip("[]").replace("'", "").replace('"', '').split(",")
        matches = [m.strip() for m in matches if m.strip()]
        for word in matches:
            if word in extra_matched:
                detailed_rows.append({
                    'word': word,
                    'model': row['model'],
                    'temperature': row['temperature'],
                    'gender_match': row['gender_match'],
                    'matched_in': 'male' if match_col == 'male_matches' else 'female',
                    'noun_gender': row['noun_gender'],
                    'adjective_gender': row['adjective_gender']
                })

# Convert to DataFrame
detailed_df = pd.DataFrame(detailed_rows)

# Group in detail
grouped = (
    detailed_df
    .groupby(['word', 'model', 'temperature', 'noun_gender', 'adjective_gender'])
    .agg(
        male_count=('matched_in', lambda x: (x == 'male').sum()),
        female_count=('matched_in', lambda x: (x == 'female').sum()),
        aligned_count=('gender_match', lambda x: x.sum()),
        nonaligned_count=('gender_match', lambda x: (~x).sum()),
    )
    .reset_index()
)

# Add totals and dominant_gender
grouped['total'] = grouped['male_count'] + grouped['female_count']
grouped['dominant_gender'] = grouped.apply(
    lambda row: 'male' if row['male_count'] > row['female_count']
    else ('female' if row['female_count'] > row['male_count'] else 'tie'),
    axis=1
)

# Sort and save
grouped = grouped.sort_values(by='total', ascending=False)
grouped.to_csv(Path(OUTPUT_FOLDER) / "extra_words_full_grouped.csv", index=False)
print(grouped.head(10))
