#!/usr/bin/env python3
"""def of a matrix"""
import numpy as np


def definiteness(matrix):
    """definiteness calculation"""
    if len(matrix) == 0:
        return None
    if type(matrix) != np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    if not np.array_equal(matrix, matrix.T):
        return None
    if np.all(np.linalg.eigvals(matrix) > 0):
        return("Positive definite")
    elif np.all(np.linalg.eigvals(matrix) < 0):
        return("Negative definite")
    elif np.all(np.linalg.eigvals(matrix) >= 0):
        return("Positive semi-definite")
    elif np.all(np.linalg.eigvals(matrix) <= 0):
        return("Negative semi-definite")
    return("Indefinite")
