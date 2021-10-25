#!/usr/bin/env python3
"""Gaussian distro"""
import numpy as np


class MultiNormal:
    """multivar normal proba"""
    def __init__(self, data):
        """init function"""
        if type(data) != np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        d, n = data.shape
        self.mean = data.mean(axis=1, keepdims=True)
        devi = data - self.mean
        self.cov = np.matmul(devi, devi.T,) / (n - 1)
