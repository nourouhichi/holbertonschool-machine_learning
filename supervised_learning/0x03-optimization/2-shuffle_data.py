#!/usr/bin/env python3
"""new module"""

import numpy as np


def shuffle_data(X, Y):
    """shuffle function"""
    order = np.random.permutation(X.shape[0])
    print(x)
    return X[order], Y[order]
