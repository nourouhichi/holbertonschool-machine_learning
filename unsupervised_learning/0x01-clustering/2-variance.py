#!/usr/bin/env python3
"""Variance"""
import numpy as np


def variance(X, C):
    """total intra-cluster avriance"""
    dist = np.argmin(np.linalg.norm((X[
        :, np.newaxis, :] - C), axis=2), axis=1)
    return np.linalg.norm(X - C[dist]) ** 2
