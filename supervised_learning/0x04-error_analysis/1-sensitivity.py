#!/usr/bin/env python3
"""new module"""

import numpy as np


def sensitivity(confusion):
    """sensitivity function"""
    sum = 0
    sens = np.zeros((confusion.shape[0], ))
    for i in range(confusion.shape[0]):
        for y in range(confusion.shape[0]):
            sum += confusion[i][y]
        sens[i] = confusion[i][i] / (sum)
        sum = 0
    return sens
