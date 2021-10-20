#!/usr/bin/env python3
"""determinant advanced linear algb"""


def determinant(matrix):
    if matrix == [[]]:
        return 1
    if type(matrix) != list or matrix == []:
        raise TypeError('matrix must be a list of lists')
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a square matrix')
    for i in matrix:
        if type(i) is not list:
            raise TypeError('matrix must be a list of lists')
    ind = list(range(len(matrix)))
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    else:
        det = 0
        for i in ind:
            focus = matrix.copy()
            focus = focus[1:]
            h = len(focus)
            for j in range(h):
                focus[j] = focus[j][0:i] + focus[j][i+1:]
            det += (-1) ** (i % 2) * matrix[0][i] * determinant(focus)
    return det
