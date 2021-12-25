#!/usr/bin/env python3
"""sdp attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """sdp attention calculation"""
    mult = tf.matmul(Q, K, transpose_b=True)
    x = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = mult / tf.math.sqrt(x)
    if mask is not None:
        scaled += (mask * -1e9)
    softmax = tf.nn.softmax(scaled, axis=-1)
    return tf.matmul(softmax, V), softmax
