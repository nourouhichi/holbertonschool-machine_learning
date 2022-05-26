#!/usr/bin/env python3
"""policy gradient module"""
import numpy as np


def policy(matrix, weight):
    """policy function on given weight"""
    exp = np.exp(matrix.dot(weight))
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """monte carlo policy gradient functioon"""
    pol = policy(state, weight)
    action = np.random.choice(len(pol[0]), p=pol[0])
    reshaped = pol.reshape(-1, 1)
    softmax = (np.diagflat(reshaped) - np.dot(reshaped, reshaped.T))[action, :]
    log = softmax / pol[0, action]
    gradient = state.T @ log[None, :]
    return action, gradient
