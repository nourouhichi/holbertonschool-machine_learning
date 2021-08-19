#!/usr/bin/env python3
"""rmsprop"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """returns tensor"""
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
