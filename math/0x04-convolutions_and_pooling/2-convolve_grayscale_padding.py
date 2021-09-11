#!/usr/bin/env python3
"""new module"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding[0], padding[1]
    new_h = h + 2 * ph - kh + 1
    new_w = w + 2 * pw - kw + 1
    output = np.zeros((m, h, w))
    img_pad = np.pad(
        array=images,
        pad_width=((0,), (ph,), (pw,)),
        mode="constant",
        constant_values=0)

    for x in range(new_h):
        for y in range(new_w):
            output[:, x, y] = (img_pad[:, x:kh+x, y:kw+y] * kernel
                               ).sum(axis=(1, 2))
    return output
