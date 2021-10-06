#!/usr/bin/env python3
""" Normalization """

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """training """
    m = np.mean(Z, axis=0)
    s = np.std(Z, axis=0)
    normalized = (Z - m) / np.sqrt(s ** 2 + epsilon)
    return gamma * normalized + beta
