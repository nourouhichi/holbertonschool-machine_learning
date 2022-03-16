#!/usr/bin/env python3
"""pandas df manip"""
import pandas as pd


def from_file(filename, delimiter):
    """loading data"""
    df = pd.read_csv(filename)
    return df
