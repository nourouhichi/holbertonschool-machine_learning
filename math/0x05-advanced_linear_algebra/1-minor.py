#!/usr/bin/env python3
"""minor advanced linear algb"""


def minor_det(m, i, j):
    """determinant of the slice"""
    m1 = m.copy()
    return [row[:j] + row[j+1:] for row in (m1[:i]+m1[i+1:])]


def minor(matrix):
    """minor matrix """
    if matrix == [[]]:
        raise ValueError('matrix must be a non-empty square')
    if type(matrix) != list or matrix == []:
        raise TypeError('matrix must be a list of lists')
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a non-empty square')
    for i in matrix:
        if type(i) is not list:
            raise TypeError('matrix must be a list of lists')
    if len(matrix) == 2:
        m = []
        m.append(matrix[1])
        m.append(matrix[0])
        return m
    if len(matrix) == 1:
        return 1
    minor_m = []
    for i in range(len(matrix)):
        m_m = []
        for j in range(len(matrix[0])):
            m = minor_det(matrix, i, j)
            m_m.append(m[0][0] * m[1][1] - m[1][0] * m[0][1])
        minor_m.append(m_m)
    return minor_m
