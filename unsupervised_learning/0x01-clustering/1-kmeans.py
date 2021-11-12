#!/usr/bin/env python3
"""clustering"""
import numpy as np


def initialize(X, k):
    """random centroids"""
    if type(k) != int or k <= 0:
        return None
    try:
        maxi = np.amax(X, axis=0)
        mini = np.amin(X, axis=0)
        return np.random.uniform(mini, maxi, (k, X.shape[1]))
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """iteration operation"""
    if type(iterations) != int or iterations <= 0:
        return None
    centroids = initialize(X, k)
    for _ in range(iterations):
        clusters = np.argmin(np.linalg.norm(
                             X[:, None] - centroids, axis=-1), axis=-1)
        C = np.zeros_like(centroids)
        for c in range(k):
            if c not in clusters:
                C[c] = np.random.uniform(np.amin(
                                         X, axis=0), np.amax(
                                         X, axis=0))
            else:
                C[c] = np.mean(X[clusters == c], axis=0)
        if(C == centroids).all():
            return centroids, clusters
        centroids = C
    clusters = np.argmin(np.linalg.norm(X[
                         :, None] - centroids, axis=-1), axis=-1)
    return C, clusters
