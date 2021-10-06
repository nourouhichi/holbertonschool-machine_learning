#!/usr/bin/env python3
""" Inception """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ Build cnn"""
    input = K.Input(shape=(224, 224, 3))
    l = K.layers.Conv2D(64, (7, 7), strides=(
        2, 2), padding='same', activation='relu')(input)
    l = K.layers.MaxPooling2D(
        (3, 3), strides=(
            2, 2), padding='same')(l)
    l = K.layers.Conv2D(64, (1, 1), strides=(
        1, 1), padding='same', activation='relu')(l)
    l = K.layers.Conv2D(192, (3, 3), strides=(
        1, 1), padding='same', activation='relu')(l)
    l = K.layers.MaxPooling2D(
        (3, 3), strides=(
            2, 2), padding='same')(l)
    l = inception_block(l, (64, 96, 128, 16, 32, 32))
    l = inception_block(l, (128, 128, 192, 32, 96, 64))
    l = K.layers.MaxPooling2D(
        (3, 3), strides=(
            2, 2), padding='same')(l)
    l = inception_block(l, (192, 96, 208, 16, 48, 64))
    l = inception_block(l, (160, 112, 224, 24, 64, 64))
    l = inception_block(l, (128, 128, 256, 24, 64, 64))
    l = inception_block(l, (112, 144, 288, 32, 64, 64))
    l = inception_block(l, (256, 160, 320, 32, 128, 128))
    l = K.layers.MaxPooling2D(
        (3, 3), strides=(
            2, 2), padding='same')(l)
    l = inception_block(l, (256, 160, 320, 32, 128, 128))
    l = inception_block(l, (384, 192, 384, 48, 128, 128))
    l = K.layers.AveragePooling2D((7, 7), strides=1)(l)
    l = K.layers.Dropout(0.4)(l)
    output = K.layers.Dense(1000, activation='softmax')(l)
    model = K.models.Model(inputs=input, outputs=output)
    return model
