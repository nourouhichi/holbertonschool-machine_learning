#!/usr/bin/env python3
""" Inception cnn"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """inception model"""
    inception3a = [64, 96, 128, 16, 32, 32]
    inception3b = [128, 128, 192, 32, 96, 64]
    inception4a = [192, 96, 208, 16, 48, 64]
    inception4b = [160, 112, 224, 24, 64, 64]
    inception4c = [128, 128, 256, 24, 64, 64]
    inception4d = [112, 144, 288, 32, 64, 64]
    inception4e = [256, 160, 320, 32, 128, 128]
    inception5a = [256, 160, 320, 32, 128, 128]
    inception5b = [384, 192, 384, 48, 128, 128]
    X = K.Input(shape=(224, 224, 3))
    c1 = K.layers.Conv2D(
                         64,
                         (7, 7),
                         strides=(2, 2),
                         padding="same",
                         activation="relu")(X)
    p1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding="same")(c1)
    c = K.layers.Conv2D(64, (1, 1), padding='same',
                        activation='relu')(p1)
    c2 = K.layers.Conv2D(
                         192,
                         (3, 3),
                         padding="same",
                         activation="relu")(c)
    p2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding="same")(c2)
    c3 = inception_block(p2, inception3a)
    c4 = inception_block(c3, inception3b)
    p3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding="same")(c4)
    c5 = inception_block(p3, inception4a)
    c6 = inception_block(c5, inception4b)
    c7 = inception_block(c6, inception4c)
    c8 = inception_block(c7, inception4d)
    c9 = inception_block(c8, inception4e)
    p4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding="same")(c9)
    c10 = inception_block(p4, inception5a)
    c11 = inception_block(c10, inception5b)
# flat
    p6 = K.layers.AveragePooling2D(pool_size=(7, 7))(c11)
    drop = K.layers.Dropout(0.4)(p6)
    output = K.layers.Dense(1000, activation='softmax')(drop)
    return K.models.Model(inputs=X, outputs=output)
