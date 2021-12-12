#!/usr/bin/env python3
"""recurrent neural network"""
import numpy as np


class RNNCell:
    """initiates an rnn"""
    def __init__(self, i, h, o):
        """Creates the public instance attributes
        that represent the weights and biases of the cell"""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh, self.by = np.zeros((1, h)), np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward propagation of an rnn"""
        data = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(data, self.Wh) + self.bh)
        x = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(x - np.max(x))
        y = exp / exp.sum(axis=1, keepdims=True)
        return h_next, y
