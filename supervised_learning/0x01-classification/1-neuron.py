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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def w(self):
        """getter func"""
        return self.__W

    @property
    def b(self):
        """getter func"""
        return self.__b

    @property
    """getter func"""
    def A(self):
        return self.__A
