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
        """ Calculate the cost of the model using logistic regression """
        m = Y.shape[1]
        B = np.transpose(A)
        D = np.transpose(np.log(1.0000001 - A))
        c = np.squeeze(-1 / m * (np.dot(Y, np.log(B)) + np.dot(1 - Y, D)))
        return c

    def evaluate(self, X, Y):
        """evaluation function"""
        a = self.forward_prop(X)
        prediction = np.where(a[1] >= 0.5, 1, 0)
        return prediction, self.cost(Y, a[1])

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """gradient function"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dz1 = np.dot(self.W2.T, dz2) * A1 * (1 - A1)
        self.__W2 += - alpha * np.dot(dz2, A1.T) / m
        self.__b2 += - alpha * np.sum(dz2, axis=1) / m
        self.__W1 += - alpha * np.dot(dz1, X.T) / m
        self.__b1 += - alpha * np.sum(dz1, axis=1, keepdims=True) / m
