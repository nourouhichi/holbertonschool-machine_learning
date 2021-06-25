#!/usr/bin/env python3
"""new module"""

import numpy as np


def moving_average(data, beta):
    """ wieghted moving av func"""
    wma = []
    j = 0
    for i in range(len(data)):
        j = beta * j + (1 - beta) * data[i]
        x = j / (1 - beta ** (i + 1))
        wma.append(x)
    return wma
