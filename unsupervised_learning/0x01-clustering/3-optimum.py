#!/usr/bin/env python3
"""Variance"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """optimum number of clusters"""
    try:
        if kmax == None:
            kmax = X.shape[0]
        results = []
        d_vars = []
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            results.append((C, clss))
            if k == kmin:
                small = variance(X, C)
            d_vars.append(small - variance(X, C))
        return results, d_vars
    except Exception:
        return None, None
