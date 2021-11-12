#!/usr/bin/env python3
"""markov chain"""
import numpy as np


def markov_chain(P, s, t=1):
    if type(P) != np.ndarray or type(
            s) != np.ndarray or type(
            t) != int or t < 0:
        return None
    if len(P.shape) != 2 or len(s.shape) != 2:
        return None
    if P.shape[0] != P.shape[1] or P.shape[0] != s.shape[1]:
        return None
    markov = np.copy(s)
    for i in range(t):
        markov = np.matmul(markov, P)
    return markov
