#!/usr/bin/env python3
""" theoreme de bayes """
import numpy as np


def likelihood(x, n, P):
    """likelihood calculation function"""
    if type(n) != int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) != int or x < 0:
        error = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(error)
    if n < x:
        raise ValueError('x cannot be greater than n')
    if type(P) != np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if (P < 0).any() or (P > 1).any():
        raise ValueError('All values in P must be in the range [0, 1]')
    f = np.math.factorial(n) / (np.math.factorial(x)
                                * np.math.factorial(n - x))
    return f * (P ** x) * ((1 - P) ** (n - x))
