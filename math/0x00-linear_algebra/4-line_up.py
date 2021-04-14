#!/usr/bin/env python3
"""
module
"""


def add_arrays(arr1, arr2):
    """add"""
    if len(arr1) is not len(arr2):
        return None
    return [x + y for x, y in zip(arr1, arr2)]
