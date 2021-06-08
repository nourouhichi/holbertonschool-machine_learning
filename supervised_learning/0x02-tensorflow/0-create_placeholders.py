#!/usr/bin/env python3
"""new module"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """function 1"""
    x = tf.placeholder(
        tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(
        tf.float32, shape=(None, classes), name="y")
    return x, y
