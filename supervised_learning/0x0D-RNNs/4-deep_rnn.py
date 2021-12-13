#!/usr/bin/env python3
"""recurrent neural network"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ forward propagation for a deep RNN"""
    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros(shape=(t + 1, l,  m, h))
    o = rnn_cells[-1].by.shape[1]
    Y = np.zeros(shape=(t, m, o))
    H[0] = h_0
    for time in range(t):
        x_prev = X[time]
        for layer in range(l):
            cell = rnn_cells[layer]
            x_t = x_prev
            h_prev = H[time, layer]
            h_next, y = cell.forward(h_prev, x_t)
            x_prev = h_next
            H[time + 1, layer, :, :] = x_prev
        Y[time] = y
    return H, Y
