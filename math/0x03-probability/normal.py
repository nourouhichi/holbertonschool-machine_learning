#!/usr/bin/env python3
"""module"""


class Normal:
    """normal distro"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """ init fuc """
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = float(sum(data) / len(data))
                y = 0
                for i in data:
                    y += (i - self.mean) ** 2
                self.stddev = (y / len(data)) ** (1/2)

    def z_score(self, x):
        """calc z score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calc x value of a z score"""
        return z * self.stddev + self.mean
