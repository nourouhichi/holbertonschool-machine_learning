#!/usr/bin/env python3
""" Normalization"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """training"""
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        name="layer")(prev)
    mean, v = tf.nn.moments(layer, axes=[0])
    gamma = tf.ones([n])
    beta = tf.zeros([n])
    normalized = tf.nn.batch_normalization(
        layer, mean, v, beta, gamma, 1e-8)
    return activation(normalized)
