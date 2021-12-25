#!/usr/bin/env python3
"""transformer model"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """positional encoding calculation"""
    p_embeddings = np.zeros((max_seq_len, dm))
    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            # sin for even
            p_embeddings[i, j] = np.sin(
                i / np.power(10000, (2 * j // 2) / dm))
            # cos for odd
            p_embeddings[i, j + 1] = np.cos(
                i / np.power(10000, (2 * j // 2) / dm))
    return p_embeddings
