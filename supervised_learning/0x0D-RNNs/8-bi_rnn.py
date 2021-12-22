#!/usr/bin/env python3
"""recurrent neural network"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """forward prop"""
    t, _, _ = X.shape
    F, B = [], []
    h_next = h_t
    h_prev = h_0
    for time in range(t):
        F.append(bi_cell.forward(h_prev, X[time]))
        h_prev = F[time]
        B.append(bi_cell.backward(h_next, X[t-time - 1]))
        h_next = B[time]
    F = np.array(F)
    B.reverse()
    B = np.array(B)
    H = np.concatenate((F, B), axis=2)
    return H, bi_cell.output(H)
