#!/usr/bin/env python3
"""new module"""

import numpy as np


def precision(confusion):
    """percision function"""
    sum = 0
    pre = np.zeros((confusion.shape[0], ))
    for i in range(confusion.shape[0]):
        for y in range(confusion.shape[0]):
            sum += confusion[y][i]
        pre[i] = confusion[i][i] / (sum)
        sum = 0
    return pre
