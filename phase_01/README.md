# Gender Bias in Dutch Word Embeddings

This project analyzes gender bias in Dutch word embeddings using FastText and Word2Vec models.

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the required data files:
   - FastText Dutch embeddings: [cc.nl.300.vec.gz](https://fasttext.cc/docs/en/crawl-vectors.html)
   - Dutch Word2Vec Sonar embeddings: [sonar-320.txt]
   - Dutch Corpus CSV file with adjectives

   Place these files in a `data/` directory in the project root.

4. Run the analysis:
   ```
   python main.py
   ```

## Project Structure

- `main.py`: Main script to run the complete analysis
- `config.py`: Configuration settings and file paths
- `models.py`: Functions to load word embedding models
- `preprocessing.py`: Functions to extract and filter adjectives
- `bias_metrics.py`: Functions to compute bias metrics
- `utils.py`: Helper functions for cosine similarity and bias computation
- `visualization.py`: Functions for creating plots

## Analysis Overview

1. Load Dutch word embedding models (FastText and Word2Vec)
2. Extract adjectives from a Dutch language corpus
3. Compute gender bias using cosine similarity
4. Perform permutation tests for statistical significance
5. Compare bias across both embedding models
6. Visualize results with scatter plots and bar charts

## Output

The analysis generates the following visualizations:
- Scatter plot comparing bias across models with significance categories
- Bar plots showing top male-biased and female-biased adjectives