#!/usr/bin/env python3
""" indetity cnn"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """building identity block"""
    init = K.initializers.he_normal()
    c1 = K.layers.Conv2D(
                         filters[0],
                         (1, 1),
                         padding='same',
                         kernel_initializer=init)(A_prev)
    norm1 = K.layers.BatchNormalization(axis=-1)(c1)
    l1 = K.layers.Activation('relu')(norm1)
    c2 = K.layers.Conv2D(
                         filters[1],
                         (3, 3),
                         padding='same',
                         kernel_initializer=init)(l1)
    norm2 = K.layers.BatchNormalization(axis=-1)(c2)
    l2 = K.layers.Activation('relu')(norm2)
    c3 = K.layers.Conv2D(
                         filters[2],
                         (1, 1),
                         padding='same',
                         kernel_initializer=init)(l2)
    norm3 = K.layers.BatchNormalization(axis=-1)(c3)
    output = K.layers.Concatenate()([norm3, A_prev])
    l3 = K.layers.Activation('relu')(output)

    return l3
