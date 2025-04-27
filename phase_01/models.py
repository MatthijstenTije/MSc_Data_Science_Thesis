import logging
import gensim
from gensim.models import KeyedVectors
from wefe.word_embedding_model import WordEmbeddingModel

def load_fasttext_model(file_path):
    """Load FastText embeddings using Gensim."""
    logging.info(f"Loading FastText embeddings from file: {file_path}")
    nl_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
        file_path,
        binary=False
    )
    logging.info("FastText embeddings loaded.")
    
    # Convert to WEFE-compatible format
    fasttext_model = WordEmbeddingModel(nl_embeddings, "Dutch FastText")
    logging.info("FastText Embeddings Model Created.")
    
    return fasttext_model

def load_word2vec_model(file_path):
    """Load Word2Vec model from file."""
    logging.info(f"Loading Word2Vec model from file: {file_path}")
    model_w2v = KeyedVectors.load_word2vec_format(file_path, binary=False)
    logging.info("Word2Vec Model loaded successfully.")
    
    return model_w2v

