#!/usr/bin/env python3
"""module"""


def np_elementwise(mat1, mat2):
    """elementwise"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mult = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mult, div
