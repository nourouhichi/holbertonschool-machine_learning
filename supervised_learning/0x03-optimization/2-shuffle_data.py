#!/usr/bin/env python3
"""new module"""

import numpy as np


def shuffle_data(X, Y):
    """shuffle function"""
    return np.random.permutation(X), np.random.permutation(Y)
