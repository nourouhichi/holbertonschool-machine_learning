#!/usr/bin/env python3
"""gaussian process """
import numpy as np


class GaussianProcess:
    """gaussian process """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """init function"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """kernel calculator function"""
        sq = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
        sq += (-2 * np.dot(X1, X2.T))
        K = (self.sigma_f ** 2) * np.exp((-1 / (2 * self.l ** 2)) * sq)
        return K

    def predict(self, X_s):
        """predicts mean and stdev"""
        K = self.kernel(self.X, X_s)
        s = self.kernel(X_s, X_s)
        inv = np.linalg.inv(self.K)
        mu = K.T.dot(inv).dot(self.Y).reshape(-1)
        cov = s - K.T.dot(inv).dot(K)
        cov = cov.diagonal()
        return mu, cov

    def update(self, X_new, Y_new):
        """updating"""
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
