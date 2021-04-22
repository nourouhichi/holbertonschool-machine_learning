#!/usr/bin/env python3
"""module"""


def poly_derivative(poly):
    """function"""
    if poly == [] or type(poly) is not list:
        return None
    if len(poly) == 1:
        return poly[0]
    deriv = []
    for index in range(1,len(poly)):
        deriv.append(index * poly[index])
    return deriv
