#!/usr/bin/env python3
""" Attention """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """multi head attention class """
    def __init__(self, dm, h):
        """init function"""
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """call function"""
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        p = [0, 2, 1, 3]
        param = (batch_size, -1, self.h, self.depth)
        q = tf.reshape(q, param)
        k = tf.reshape(k, param)
        v = tf.reshape(v, param)
        softmax, output = sdp_attention(q, k, v, mask)
        concat = tf.reshape(softmax, (batch_size, -1, self.dm))
        output = self.linear(concat)
        return output, output
