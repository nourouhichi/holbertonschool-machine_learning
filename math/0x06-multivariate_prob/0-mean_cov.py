#!/usr/bin/env python3
"""Gaussian distro"""
import numpy as np


def mean_cov(X):
    """mean cov"""
    if len(X.shape) != 2 or type(X) != np.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    N ,d = X.shape
    mean = np.mean(X, axis=0).reshape(1, d)
    devi = X - mean
    cov = np.matmul(devi.T, devi) / (N - 1)
    return mean, cov
