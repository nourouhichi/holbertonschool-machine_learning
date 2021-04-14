#!/usr/bin/env python3


def cat_matrices2D(mat1, mat2, axis=0):
    """concate matrices"""
    mat = []
    if axis == 1:
        for x in range(len(mat1)):
            mat.append(mat1[x] + mat2[x])
    elif axis == 0:
        mat = mat1 + mat2
    return mat
