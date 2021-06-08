#!/usr/bin/env python2
"""new module"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """acc function"""
    return tf.math.reduce_mean(
        tf.cast(
            tf.math.equal(y, y_pred),
            tf.float32))
