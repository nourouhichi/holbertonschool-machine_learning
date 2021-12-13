#!/usr/bin/env python3
"""recurrent neural network"""
import numpy as np


class LSTMCell:
    """initiates an rnn"""
    def __init__(self, i, h, o):
        """Creates the public instance attributes
        that represent the weights and biases of the cell"""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))

    def forward(self, h_prev, c_prev, x_t):
        """forward propagation of a lstm"""
        data = np.concatenate((h_prev, x_t), axis=1)
        cand = np.tanh(np.matmul(data, self.Wc) + self.bc)
        forget_gate = np.matmul(data, self.Wf) + self.bf
        forget_gate = 1 / (1 + np.exp(-forget_gate))
        output_gate = np.matmul(data, self.Wo) + self.bo
        output_gate = 1 / (1 + np.exp(-output_gate))
        update_gate = np.matmul(data, self.Wu) + self.bu
        update_gate = 1 / (1 + np.exp(-update_gate))

        c_next = update_gate * cand + forget_gate * c_prev
        h_next = output_gate * np.tanh(c_next)

        x = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return h_next, c_next, y
