#!/usr/bin/env python3
"""new module"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """acc function"""
    return tf.math.reduce_mean(
        tf.cast(
            tf.math.equal(
                tf.math.argmax(
                    y,
                    axis=-1),
                tf.math.argmax(
                    y_pred,
                    axis=-1)),
            tf.float32))
