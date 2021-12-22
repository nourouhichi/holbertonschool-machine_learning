#!/usr/bin/env python3
"""recurrent neural network"""
import numpy as np


class BidirectionalCell:
    """bidirectional rnn"""
    def __init__(self, i, h, o):
        """init function for  cell"""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))

    def forward(self, h_prev, x_t):
        """forward for bidi rnn cell"""
        data = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(data, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """backward direction of bi cell"""
        data = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(data, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """prediction of a cell"""
        x = np.matmul(H, self.Wy) + self.by
        exp = np.exp(x)
        y = exp / exp.sum(axis=2, keepdims=True)
        return y
