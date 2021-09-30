#!/usr/bin/env python3
"""covnet module"""


import numpy as np


def pool_forward(A_prev, kernel_shape,
                 stride=(1, 1), mode='max'):
    """pooling layer"""
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    output_h = int(((h - kh) / sh) + 1)
    output_w = int(((w - kw) / sw) + 1)
    output = np.zeros((m, output_h, output_w, c))
    for x in range(output_h):
        for y in range(output_w):
            if mode == "max":
                output[:, x, y, :] = np.max(
                    A_prev[:, x * sh:kh + x * sh, y * sw:kw + y * sw, :],
                    axis=(1, 2))
            if mode == "avg":
                output[:, x, y, :] = np.average(
                    A_prev[:, x*sh:kh + x*sh, y*sw:kw + y*sw, :],
                    axis=(1, 2))
    return output
