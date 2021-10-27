#!/usr/bin/env python3
""" dim reduction"""
import numpy as np


def pca(X, ndim):
    """pca function"""
    mean_sub = X - np.mean(X, axis=0)
    u, sig, v = np.linalg.svd(mean_sub)
    return np.matmul(mean_sub, v.T[:, :ndim])
