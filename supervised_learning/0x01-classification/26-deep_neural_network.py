#!/usr/bin/env python3
"""
deep Neuralnetwork  module
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle as pk


class DeepNeuralNetwork:
    """ new class"""
    def __init__(self, nx, layers):
        """init function"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights['W1'] = \
                    np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], layers[i - 1]) * \
                    np.sqrt(2 / layers[i - 1])
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """ activation function"""
        self.__cache["A0"] = X
        for i in range(self.__L):
            suc = str(i + 1)
            ln = str(i)
            z = np.matmul(self.__weights["W" + suc], self.__cache["A" + ln]) +\
                self.__weights["b" + suc]
            self.__cache["A" + str(suc)] = 1 / (1 + np.exp(- z))
        return self.cache["A" + str(i + 1)], self.__cache

    def cost(self, Y, A):
        """loss function """
        m = Y.shape[1]
        z = 1.0000001 - A
        L = (np.sum(Y * np.log(A) + (1 - Y) * np.log(z)))/m
        return (-1 * L)

    def evaluate(self, X, Y):
        """evaluation function"""
        self.forward_prop(X)
        prediction = np.where(self.__cache["A" + str(self.__L)] < 0.5, 0, 1)
        return prediction, self.cost(Y, self.__cache["A" + str(self.__L)])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """gradient function"""
        m = Y.shape[1]
        dz = cache["A" + str(self.__L)] - Y
        for i in range(self.L, 0, -1):
            A = cache["A" + str(i-1)]
            dW = np.matmul(dz, A.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            dz = np.matmul(self.__weights[
                "W" + str(i)].T, dz) * (A * (1 - A))
            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db

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
        axey = []
        axex = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if verbose and i % 100 == 0:
                print("Cost after {} iterations: {}".format(
                    i, self.cost(Y, A)))
                axey.append(self.cost(Y, A))
                axex.append(i)
        if graph:
            plt.plot(axex, axey, color="blue")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ pickling"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
            with open(filename, "wb") as f:
                pk.dump(self, f)

    @staticmethod
    def load(filename):
        """unpickling"""
        try:
            with open(filename, "rb") as f:
                load = pk.load(f)
            return load
        except Exception:
            return None
