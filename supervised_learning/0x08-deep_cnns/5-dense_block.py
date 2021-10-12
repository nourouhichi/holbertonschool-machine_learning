#!/usr/bin/env python3
""" Dense Block modules"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """dense block """
    out = X
    init = K.initializers.he_normal()
    for i in range(layers):
        norm1 = K.layers.BatchNormalization()(out)
        l1 = K.layers.Activation('relu')(norm1)
        c1 = K.layers.Conv2D(4 * growth_rate,
                             (1, 1),
                             kernel_initializer=init,
                             padding='same')(l1)
        norm2 = K.layers.BatchNormalization()(c1)
        l2 = K.layers.Activation('relu')(norm2)
        c2 = K.layers.Conv2D(
                             growth_rate,
                             (3, 3),
                             kernel_initializer=init,
                             padding='same')(l2)
        nb_filters += growth_rate
        out = K.layers.Concatenate()([out, c2])
    return out, nb_filters
