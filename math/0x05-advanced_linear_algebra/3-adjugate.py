#!/usr/bin/env python3
"""cofactor of a matrix"""


def determinant(matrix):
    """det matrix calculation"""
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


def minor(matrix):
    """minor matrix """
    if type(matrix) != list or matrix == []:
        raise TypeError('matrix must be a list of lists')
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a non-empty square matrix')
    if matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    for i in matrix:
        if type(i) is not list:
            raise TypeError('matrix must be a list of lists')
    if len(matrix) == 1:
        return [[1]]
    minor_m = []
    for i in range(len(matrix)):
        m_m = []
        for j in range(len(matrix[0])):
            m = [y[:] for y in matrix]
            del m[i]
            for z in m:
                del z[j]
            det = determinant(m)
            m_m.append(det)
        minor_m.append(m_m)
    return minor_m


def cofactor(matrix):
    """cofactor calculator"""
    m = minor(matrix)
    for i in range(len(m)):
        for j in range(len(m[0])):
            if(i + j) % 2 != 0:
                m[i][j] *= (-1)
    return m


def adjugate(matrix):
    """adj calculator"""
    cof = cofactor(matrix)
    ad = []
    for i in range(len(cof)):
        inv = []
        for j in range(len(cof)):
            inv.append(cof[j][i])
        ad.append(inv)
    return ad
