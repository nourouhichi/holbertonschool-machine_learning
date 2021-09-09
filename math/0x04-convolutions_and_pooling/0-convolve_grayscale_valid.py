#!/usr/bin/env python3
"""new module"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """convolving a grayscale image"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    xOutput = h - kh + 1
    yOutput = w - kw + 1
    output = np.zeros((m, xOutput, yOutput))
    for x in range(xOutput):
        for y in range(yOutput):
            output[:, x, y] = (kernel * images[
                            :, x: x + kh, y: y + kw]).sum(axis=(1, 2))
    return(output)
