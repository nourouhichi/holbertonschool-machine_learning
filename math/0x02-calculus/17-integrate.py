#!/usr/bin/env python3
"""module"""


def poly_integral(poly, C=0):
    """function"""
    if poly == [] or type(poly) is not list or not isinstance(C, (float, int)):
        return None
    prim = [C]
    if len(poly) == 1 and poly[0] == 0:
        return prim
    for index in range(len(poly)):
        if not isinstance(poly[index], (int, float)):
            return None
        if (poly[index] / (index + 1)) % 1 == 0:
            prim.append(int(poly[index] / (index + 1)))
        else:
            prim.append(poly[index] / (index + 1))
    return prim
