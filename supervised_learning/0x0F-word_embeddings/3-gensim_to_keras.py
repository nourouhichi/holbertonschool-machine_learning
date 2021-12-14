#!/usr/bin/env python3
"""nlp word embeddings"""
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """converts a gensim word2vec model to a
    keras Embedding layer"""
    return model.wv.get_keras_embedding(True)
