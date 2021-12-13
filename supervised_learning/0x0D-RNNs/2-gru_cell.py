#!/usr/bin/env python3
"""recurrent neural network"""
import numpy as np


class GRUCell:
    """initiates an rnn"""
    def __init__(self, i, h, o):
        """Creates the public instance attributes
        that represent the weights and biases of the cell"""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh, self.by = np.zeros((1, h)), np.zeros((1, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))

    def forward(self, h_prev, x_t):
        """forward propagation of a gru"""
        data = np.concatenate((h_prev, x_t), axis=1)
        reset_gate = np.matmul(data, self.Wr) + self.br
        reset_gate = 1 / (1 + np.exp(-reset_gate))
        update_gate = np.matmul(data, self.Wz) + self.bz
        update_gate = 1 / (1 + np.exp(-update_gate))

        conc = np.concatenate(((reset_gate * h_prev), x_t), axis=1)
        cand = np.tanh(np.matmul(conc, self.Wh) + self.bh)
        h_next = (1 - update_gate) * h_prev + update_gate * cand

        x = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return h_next, y
