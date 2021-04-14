#!/usr/bin/env python3


def matrix_shape(matrix):
    """shape"""
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])
