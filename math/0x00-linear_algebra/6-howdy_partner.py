#!/usr/bin/env python3
"""
module
"""


def cat_arrays(arr1, arr2):
    """concate"""
    arr = arr1.copy()
    for i in arr2:
        arr.append(i)
    return arr
