#!/usr/bin/env python3
"""new module"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """pooling with many channels"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    output_h = int(((h - kh) / sh) + 1)
    output_w = int(((w - kw) / sw) + 1)
    output = np.zeros((m, output_h, output_w, c))
    for x in range(output_h):
        for y in range(output_w):
            if mode == "max":
                output[:, x, y, :] = np.max(
                    images[:, x * sh:kh + x * sh, y * sw:kw + y * sw, :],
                    axis=(1, 2))
            if mode == "avg":
                output[:, x, y, :] = np.average(
                    images[:, x*sh:kh + x*sh, y*sw:kw + y*sw, :],
                    axis=(1, 2))
    return output
