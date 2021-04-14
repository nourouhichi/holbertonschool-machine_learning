#!/usr/bin/env python3
"""
module
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concate matrices"""
    if axis == 1:
        if len(mat1) == len(mat2):
            return [x + y for x, y in zip(mat1, mat2)]
        else:
            return None
    elif axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            return [x.copy() for x in mat1] + [x.copy() for x in mat2]
        else:
            return None
