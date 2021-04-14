#!/usr/bin/env python3
"""
module
"""


def add_matrices2D(mat1, mat2):
    """add matrix"""
    if(len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0])):
        mat = []
        for x in range(len(mat1)):
            submat = []
            for y in range(len(mat1[x])):
                submat.append(mat1[x][y] + mat2[x][y])
            mat.append(submat)
        return mat
