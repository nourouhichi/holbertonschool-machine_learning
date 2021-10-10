#!/usr/bin/env python3
""" Convolutional Neural Networks """
import numpy as np
from numpy.core.numeric import zeros_like


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """backpropagation over a pooling layer"""
    m, h, w, cnew = A_prev.shape
    m, hnew, wnew, cnew = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    A = A_prev
    da_prev = np.zeros_like(A_prev)
    for z in range(m):
        for x in range(hnew):
            for y in range(wnew):
                for c in range(cnew):
                    if mode == "max":
                        slice = A[z, x*sh:x*sh+kh, y*sw:y*sw+kw, c]
                        bool = (np.max(slice) == slice)
                        da_prev[z, x*sh:x*sh+kh, y*sw:y *
                                sw+kw, c] += np.multiply(bool, dA[z, x, y, c])
                    if mode == 'avg':
                        da_prev[z, x * sh:x * sh + kh, y *
                                sw:y * sw + kw, c] += dA[z, x, y, c] / kh / kw
    return da_prev
