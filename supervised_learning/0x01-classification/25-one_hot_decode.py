#!/usr/bin/env python3
"""new module"""


import numpy as np
from numpy.core.fromnumeric import shape


def one_hot_decode(one_hot):
    """decoding"""
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot.T, axis=1)
