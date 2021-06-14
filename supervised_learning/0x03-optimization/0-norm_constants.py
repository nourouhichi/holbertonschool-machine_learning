#!/usr/bin/env python3
"""new module"""

import numpy as np


def normalization_constants(X):
    """stand function"""
    return np.mean(X, axis=0), np.std(X, axis=0)
