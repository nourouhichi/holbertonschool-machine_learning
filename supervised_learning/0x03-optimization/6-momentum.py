#!/usr/bin/env python3
"""new module"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """training op"""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
