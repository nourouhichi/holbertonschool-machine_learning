#!/usr/bin/env python3
"""clustring"""
import numpy as np


def initialize(X, k):
    """random centroids"""
    if type(k) != int or k <= 0:
        return None
    maxi = np.amax(X, axis=0)
    mini = np.amin(X, axis=0)
    try:
        return np.random.uniform(mini, maxi, (k, X.shape[1]))
    except Exception:
        return None
