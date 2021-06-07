#!/usr/bin/env python3
"""new module"""


import numpy as np


def one_hot_encode(Y, classes):
    """encoding"""
    if type(Y) is not numpy.ndarray:
        return None
    try:
        one_hot_encoded = np.zeros((classes, len(Y)))
        for i in range(len(Y)):
            one_hot_encoded[Y[i], i] = 1
        return one_hot_encoded
    except Exception:
        return None
