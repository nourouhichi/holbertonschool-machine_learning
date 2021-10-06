#!/usr/bin/env python3
""" Identity Block module"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ Build an identity block """
    initializer = K.initializers.he_normal()
    l = K.layers.Conv2D(
        filters[0],
        (1,
         1),
        padding='same',
        kernel_initializer=initializer)(A_prev)
    l = K.layers.BatchNormalization()(l)
    l = K.layers.Activation('relu')(l)
    l = K.layers.Conv2D(
        filters[1],
        (3,
         3),
        padding='same',
        kernel_initializer=initializer)(l)
    l = K.layers.BatchNormalization()(l)
    l = K.layers.Activation('relu')(l)
    l = K.layers.Conv2D(
        filters[2],
        (1,
         1),
        padding='same',
        kernel_initializer=initializer)(l)
    l = K.layers.BatchNormalization()(l)
    output = K.layers.Add()([l, A_prev])
    output = K.layers.Activation('relu')(output)
    return output