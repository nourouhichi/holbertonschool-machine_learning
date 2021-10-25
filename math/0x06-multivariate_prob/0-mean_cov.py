#!/usr/bin/env python3
"""Gaussian distro"""
import numpy as np


def mean_cov(X):
    """mean cov"""
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    elif X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0)
    X -= mean
    cov = np.dot(X.T, X.conj()/(X.shape[0] - 1))
    return mean, cov