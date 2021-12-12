#!/usr/bin/env python3
"""recurrent neural network"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """forward prop"""
    t, m, i = X.shape
    o = rnn_cell.Wy.shape[1]
    m, h = h_0.shape
    H = np.zeros(shape=(t + 1, m, h))
    Y = np.zeros(shape=(t, m, o))
    H[0] = h_0
    h_prev = h_0
    for time in range(t):
        H[time + 1, :, :], Y[
            time, :, :] = rnn_cell.forward(
                h_prev, X[time, :, :])
        h_prev = H[time + 1, :, :]
    return H, Y
