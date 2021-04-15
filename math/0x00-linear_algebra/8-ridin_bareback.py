#!/usr/bin/env python3
"""module"""


def mat_mul(mat1, mat2):
    """ multiply"""
    if len(mat1[0]) != len(mat2):
        return None
    matrix = []
    mat = []
    new_mat2 = []
    for i in range(len(mat2[0])):
        new_mat2.append([row[i] for row in mat2])
    for i in mat1:
        for j in new_mat2:
            mat.append(sum([k * l for k, l in zip(i, j)]))
        matrix.append(mat)
        mat = []
    return matrix
