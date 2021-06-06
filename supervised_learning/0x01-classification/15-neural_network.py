#!/usr/bin/env python3
"""
Neuralnetwork  module
"""
import numpy as np
import matplotlib.pyplot as plt


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
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(- z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(- z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """loss function """
        m = Y.shape[1]
        z = 1.0000001 - A
        L = (np.sum(Y * np.log(A) + (1 - Y) * np.log(z)))/m
        return (-1 * L)

    def evaluate(self, X, Y):
        """evaluation function"""
        a = self.forward_prop(X)
        prediction = np.where(self.__A2 < 0.5, 0, 1)
        return prediction, self.cost(Y, self.__A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """gradient function"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dz1 = np.matmul(self.W2.T, dz2) * (A1 * (1 - A1))
        self.__W2 = self.__W2 - alpha * np.matmul(dz2, A1.T) / m
        self.__b2 = self.__b2 - alpha * np.sum(dz2, axis=1, keepdims=True) / m
        self.__W1 = self.__W1 - alpha * np.matmul(dz1, X.T) / m
        self.__b1 = self.__b1 - alpha * np.sum(dz1, axis=1, keepdims=True) / m

    def train(
            self, X, Y, iterations=5000, alpha=0.05,
            verbose=True, graph=True, step=100):
        """training function"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        x_axis = []
        y_axis = []
        for i in range(iterations + 100):
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if verbose and i % step is 0:
                print("Cost after {} iterations: {}".format(
                    i, self.cost(Y, self.__A2)))
                x_axis.append(self.cost(Y, self.__A2))
                y_axis.append(i)

        if graph:
            plt.plot(y_axis, x_axis, color="blue")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
