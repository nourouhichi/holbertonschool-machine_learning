#!/usr/bin/env python3
"""module"""


class Poisson:
    """poisson ditro class"""

    def __init__(self, data=None, lambtha=1.):
        """init  func"""
        self.data = data
        if data is None:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError("lambtha must be a positive value")
        elif type(data) is not list:
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")
        else:
            self.lambtha = float(sum(data) / len(data))
