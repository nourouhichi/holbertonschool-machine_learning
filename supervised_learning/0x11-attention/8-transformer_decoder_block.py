#!/usr/bin/env python3
"""decoder block"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """decoder class """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """init funtion"""
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ call function"""
        mh_1, _ = self.mha1(x, x, x, look_ahead_mask)
        mh_1 = self.dropout1(mh_1, training=training)
        o1 = self.layernorm1(mh_1 + x)
        mh_2, _ = self.mha2(o1, encoder_output,
                            encoder_output,
                            padding_mask)
        mh_2 = self.dropout2(mh_2, training=training)
        o2 = self.layernorm2(mh_2 + o1)
        seq = tf.keras.Sequential([self.dense_hidden, self.dense_output])
        output = seq(o2)
        output = self.dropout3(output, training=training)
        o3 = self.layernorm3(output + o2)
        return o3
