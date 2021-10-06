#!/usr/bin/env python3
""" Inception cnn"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ inception block """
    c1 = K.layers.Conv2D(
        filters[0], (1, 1), padding='same', activation='relu')(A_prev)
    c1_3 = K.layers.Conv2D(
        filters[1], (1, 1), padding='same', activation='relu')(A_prev)
    c3 = K.layers.Conv2D(
        filters[2], (3, 3), padding='same', activation='relu')(c1_3)
    c1_5 = K.layers.Conv2D(
        filters[3], (1, 1), padding='same', activation='relu')(A_prev)
    c5 = K.layers.Conv2D(
        filters[4], (5, 5), padding='same', activation='relu')(c1_5)
    pool = K.layers.MaxPooling2D(
        (3, 3), strides=(
            1, 1), padding='same')(A_prev)
    convp_1 = K.layers.Conv2D(
        filters[5], (1, 1), padding='same', activation='relu')(pool)
    output = K.layers.concatenate([c1, c3, c5, convp_1], axis=-1)
    return output
