#!/usr/bin/env python3
"""markov chain"""
import numpy as np


def absorbing(P):
    """absorbing markov chain"""
    try:
        n = P.shape[0]
        diag = np.where(np.diag(P) == 1, 1, 0)
        if not np.any(diag == 1):
            return False

        d, v = np.linalg.eig(P)
        P_ = np.matmul(np.matmul(v, np.diag(
                       d == 1).astype(int)), np.linalg.inv(v))

        if int(np.ceil(np.max(P_.sum()))) != n:
            return False
        return True

    except Exception:
        return False
