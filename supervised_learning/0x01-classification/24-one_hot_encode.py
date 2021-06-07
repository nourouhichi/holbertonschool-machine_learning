#!/usr/bin/env python3
import numpy as np


def one_hot_encode(Y, classes):
    one_hot_encoded = np.zeros((classes, len(Y)))
    for i in range(len(Y)):
        one_hot_encoded[Y[i], i] = 1
    return one_hot_encoded
