#!/usr/bin/env python3
"""new module"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """training"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
