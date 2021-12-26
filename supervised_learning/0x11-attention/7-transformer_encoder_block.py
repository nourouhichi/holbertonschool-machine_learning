#!/usr/bin/env python3
"""'encoder of transformer"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """encoder class"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """init function"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """call function"""
        mh, _ = self.mha(x, x, x, mask)
        mh = self.dropout1(mh, training=training)
        y = self.layernorm1(x + mh)
        s = tf.keras.Sequential([self.dense_hidden, self.dense_output])
        seq_output = self.dropout2(s(y), training=training)
        return self.layernorm2(seq_output)
