#!/usr/bin/env python3
"""markov chain"""
import numpy as np


def regular(P):
    """stationary"""
    if type(P) != np.ndarray:
        return None
    if len(P.shape) != 2 or P.shape[1] != P.shape[0]:
        return None
    try:
        power = np.linalg.matrix_power(P, 50)
    except Exception:
        return None
    if np.any(power <= 0):
        return None
    return np.array([power[0]])
