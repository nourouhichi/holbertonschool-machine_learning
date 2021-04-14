#!/usr/bin/env python3
def matrix_transpose(matrix):
    """transpose"""
    mat = []

    for x in range(len(matrix[0])):
        submat = []
        for j in range(len(matrix)):
            submat.append(matrix[j][x])
        mat.append(submat)
    return mat
