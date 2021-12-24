#!/usr/bin/env python3
"""encoder for machine translation"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """encoder class"""
    def __init__(self, vocab, embedding, units, batch):
        """init function"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """initializes hidden states tensor"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """calculates output and hidden state"""
        x = self.embedding(x)
        return self.gru(x, initial)
