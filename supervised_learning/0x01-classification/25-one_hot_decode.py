#!/usr/bin/env python3
"""new module"""


import numpy as np


def one_hot_decode(one_hot):
    """decoding"""
    if type(one_hot) is not np.ndarray:
        return None
    return np.argmax(one_hot.T, axis=1)
