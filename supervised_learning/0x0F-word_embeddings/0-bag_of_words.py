#!/usr/bin/env python3
"""nlp word embeddings"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ creates a bag of words embedding matrix"""
    cv = CountVectorizer(vocabulary=vocab)
    return cv.fit_transform(sentences).toarray(), cv.get_feature_names()
