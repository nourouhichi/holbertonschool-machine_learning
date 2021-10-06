#!/usr/bin/env python3
""" Normalization"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """training"""
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
