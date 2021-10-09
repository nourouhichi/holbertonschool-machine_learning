#!/usr/bin/env python3
"""covnet module"""


import numpy as np


def conv_forward(A_prev, W, b,
                 activation, padding="same",
                 stride=(1, 1)):
    """forward prop in a convnet"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'valid':
        pad_h = 0
        pad_w = 0
    elif padding == 'same':
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    output_h = int(((h_prev - kh + 2 * pad_h) / sh) + 1)
    output_w = int(((w_prev - kw + 2 * pad_w) / sw) + 1)
    output = np.zeros((m, output_h, output_w, c_new))
    img_pad = np.pad(
                     array=A_prev,
                     pad_width=((0,), (pad_h,), (pad_w,), (0,)),
                     mode="constant",
                     constant_values=0)
    for x in range(output_h):
        for y in range(output_w):
            for c in range(c_new):
                piece = img_pad[:, x*sh:x*sh+kh, y*sw:y*sw+kw, :]
                output[:, x, y, c] = activation(
                                                piece * W[:, :, :, c] +
                                                b[:, :, :, c]).sum(
                                                axis=(1, 2, 3))

    return output
