#!/usr/bin/env python3
"""new module"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """creating a layer"""
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        name="layer",
    )
    return layer(prev)
