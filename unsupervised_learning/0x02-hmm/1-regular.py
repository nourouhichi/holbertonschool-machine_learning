#!/usr/bin/env python3
"""markov chain"""
import numpy as np


def regular(P):
    """stationary"""
    try:
        power = np.linalg.matrix_power(P, 100)
    except Exception:
        return None
    if np.any(power <= 0):
        return None
    return np.array([P[0]])
