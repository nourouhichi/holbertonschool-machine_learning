#!/usr/bin/env python3
"""
Neuralnetwork  module
"""
import numpy as np


class NeuralNetwork:
    """ new class"""
    def __init__(self, nx, nodes):
        """init function"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """ for propagation function"""
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(- z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(- z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ cost func"""
        m = Y.shape[1]
        B = np.transpose(np.log(1.000001 - A))
        D = np.log(np.transpose(A))
        c = np.squeeze(-1 / m * (np.dot(Y, D) + np.dot(1 - Y, B)))
        return c
