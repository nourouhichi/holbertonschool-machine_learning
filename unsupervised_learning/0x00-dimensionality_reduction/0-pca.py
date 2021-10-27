#!/usr/bin/env python3
""" dim reduction"""
import numpy as np


def pca(X, var=0.95):
    """feature extraction"""
    u, sig, v = np.linalg.svd(X)
    e = np.cumsum(sig) / np.sum(sig)
    i = np.where(e <= var, 1, 0)
    i = np.sum(i)
    return v.T[:, :i + 1]
