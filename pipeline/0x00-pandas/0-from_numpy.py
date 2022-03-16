#!/usr/bin/env python3
"""pandas df manip"""
import pandas as pd


def from_numpy(array):
    """creating a df from np"""
    shape = array.shape
    alp = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    return pd.DataFrame(array, columns=alp[:shape[1]])
