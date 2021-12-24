#!/usr/bin/env python3
"""encoder for machine translation"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """slefattention class"""
    def __init__(self, units):
        """init function"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """returns context and weights"""
        s_prev = tf.expand_dims(s_prev, 1)
        o = self.V(tf.nn.tanh(self.W(s_prev) + self.U(
            hidden_states)))
        weights = tf.nn.softmax(o, axis=1)
        aux = weights * hidden_states
        context = tf.reduce_sum(aux, axis=1)
        return context, weights
