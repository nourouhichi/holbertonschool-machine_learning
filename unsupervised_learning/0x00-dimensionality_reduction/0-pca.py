#!/usr/bin/env python3
""" dim reduction"""
import numpy as np


def pca(X, var=0.95):
    """feature extraction"""
    u, sig, v = np.linalg.svd(X)
    cov = np.cumsum(sig) / np.sum(sig)
    i = np.where(cov <= var, 1, 0)
    print(cov)
    print(i)
    i = np.sum(i)
    return v.T[:, :i]
