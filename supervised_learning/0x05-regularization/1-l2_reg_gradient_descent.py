#!/usr/bin/env python3
""" Reg """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """gradient descent regularization"""
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        dW = np.matmul(dz, A.T) / m + lambtha / m * weights["W" + str(i)]
        db = np.sum(
            dz, axis=1, keepdims=True) / m + lambtha / m
        dz = np.matmul(weights["W" + str(i)].T, dz) * (1 - A * A )
        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db
