#!/usr/bin/env python3
"""module"""


class Poisson:
    """poisson ditro class"""
    e = 2.7182818285

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

    def pmf(self, k):
        """calculating pmf"""
        if type(int) is not int:
            k = int(k)
        if k < 0:
            return 0
        factorial = 1
        for i in range(2, k + 1):
            factorial = factorial * i
        return (self.lambtha ** k) * (self.e ** (- self.lambtha)) / factorial

    def cdf(self, k):
        """calculate cdf"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
