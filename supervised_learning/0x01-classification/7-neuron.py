#!/usr/bin/env python3
"""
Neuron module
"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """ new class"""

    def __init__(self, nx):
        """init func"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """forward propagation func"""
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(- z))
        return self.__A

    def cost(self, Y, A):
        """cost function"""
        m = Y.shape[1]
        B = np.transpose(np.log(1.0000001 - A))
        D = np.log(np.transpose(A))
        c = np.squeeze(-1 / m * (np.dot(Y, D) + np.dot(1 - Y, B)))
        return c

    def evaluate(self, X, Y):
        """evaluation function"""
        a = self.forward_prop(X)
        prediction = np.where(a >= 0.5, 1, 0)
        return prediction, self.cost(Y, a)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """gradient function"""
        m = Y.shape[1]
        B = np.transpose(A)
        x = np.dot((A - Y), X.T) / m
        y = (np.sum(A - Y)) / m
        self.__W += - alpha * x
        self.__b += - alpha * y

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
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if verbose and i % step is 0:
                print("Cost after {} iterations: {}".format(
                    i, self.cost(Y, self.__A)))
                x_axis.append(self.cost(Y, self.__A))
                y_axis.append(i)
        if graph:
            plt.plot(y_axis, x_axis, color="blue")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
