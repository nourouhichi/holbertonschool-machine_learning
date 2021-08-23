#!/usr/bin/env python3
"""new module"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """f1 score"""
    x = precision(confusion)
    y = sensitivity(confusion)
    return 2 * ((x * y) / (x + y))
