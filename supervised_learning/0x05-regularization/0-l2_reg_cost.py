#!/usr/bin/env python3
""" Reg """

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Calculates l2 reg cost func """
    norm = 0
    for i in range(1, L + 1):
        norm += np.linalg.norm(weights["W" + str(i)]) ** 2
    return cost + (lambtha / (2 * m)) * norm
