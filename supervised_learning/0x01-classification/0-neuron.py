#!/usr/bin/env python3
"""
Neuron module
"""
import numpy as np


class Neuron:
    """ new class"""

    def __init__(self, nx):
        """init func"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
