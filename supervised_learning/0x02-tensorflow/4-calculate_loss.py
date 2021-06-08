#!/usr/bin/env python3
"""new module"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """loss calculation"""
    return tf.losses.softmax_cross_entropy(
        y,
        y_pred,
        weights=1.0,
        label_smoothing=0,
        scope=None,
    )
