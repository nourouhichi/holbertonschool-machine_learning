#!/usr/bin/env python3
"""new module"""


import numpy as np
from numpy.core.fromnumeric import shape


def one_hot_decode(one_hot):
    """decoding"""
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    x = np.zeros((len(one_hot),), dtype=int)
    for i in range(len(one_hot)):
        for y in range(len(one_hot[i])):
            if one_hot[i][y] == 1:
                x[y] = i
    return x
