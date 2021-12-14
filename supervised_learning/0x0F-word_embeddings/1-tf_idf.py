#!/usr/bin/env python3
"""nlp word embeddings"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """tf-idf embeddings"""
    vc = TfidfVectorizer(vocabulary=vocab)
    return vc.fit_transform(sentences).toarray(), vc.get_feature_names()
