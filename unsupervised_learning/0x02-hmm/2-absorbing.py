#!/usr/bin/env python3
"""markov chain"""
import numpy as np
from numpy.core.fromnumeric import shape


def absorbing(P):
    """absorbing markov chain"""
    n = P.shape[0]
    diag = P.diagonal()
    absorb = P[np.where(diag == 1)]
    trans = np.delete(P, np.where(diag == 1), axis=0)
    Q = trans[:, absorb.shape[0]:]
    R = trans[:, :absorb.shape[0]]
    I = np.zeros((absorb.shape[0], absorb.shape[0]))
    print(I)
    np.fill_diagonal(I, 1)
    try:
        print((I - Q).T * R)
        fond = (I - Q).T * R
    except Exception:
        pass
    return False
        
