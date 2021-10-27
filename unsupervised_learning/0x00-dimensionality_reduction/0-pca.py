#!/usr/bin/env python3
""" dim reduction"""
import numpy as np


def pca(X, var=0.95):
    """feature extraction"""
    x = X.copy()
    cov = np.cov(x.T)
    eigen_va, eigen_ve = np.linalg.eig(cov)
    proj = (eigen_ve.T[:][:2]).T
    return proj