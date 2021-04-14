#!/usr/bin/env python3
"""
module
"""


def add_arrays(arr1, arr2):
    """add"""
    if len(arr1) != len(arr2):
        return None
    arr = []
    for x in range(len(arr1)):
        arr.append(arr1[x] + arr2[x])
    return arr
