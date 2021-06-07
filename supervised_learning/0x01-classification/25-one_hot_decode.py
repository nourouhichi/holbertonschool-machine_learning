#!/usr/bin/env python3
"""new module"""


import numpy as np


def one_hot_decode(one_hot):
    """decoding"""
    if type(one_hot) is not np.ndarray or one_hot.ndim is  not 2:
        return None
    try:
        x = np.zeros((len(one_hot),), dtype=int)
        for i in range(len(one_hot)):
            for y in range(len(one_hot[i])):
                if one_hot[i][y] == 1:
                    x[y] = i
        return x
    except Exception:
        return None
