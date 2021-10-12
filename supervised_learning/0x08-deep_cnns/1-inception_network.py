#!/usr/bin/env python3
""" Inception cnn"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """inception model"""
    X = K.Input(shape=(224, 224, 3))
    c1 = K.layers.Conv2D(
                         64,
                         (7, 7),
                         strides=(2, 2),
                         padding="same",
                         activation="relu")(X)
    p1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding="same")(c1)
    c = K.layers.Conv2D(64, (1, 1), strides=(1, 1),
                        padding='same', activation='relu')(p1)
    c2 = K.layers.Conv2D(
                         192,
                         (3, 3),
                         padding="same",
                         activation="relu")(c)
    p2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding="same")(c2)
    c3 = inception_block(p2, [64, 96, 128, 16, 32, 32])
    c4 = inception_block(c3, [128, 128, 192, 32, 96, 64])
    p3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding="same")(c4)
    c5 = inception_block(p3, [192, 96, 208, 16, 48, 64])
    c6 = inception_block(c5, [160, 112, 224, 24, 64, 64])
    c7 = inception_block(c6, [128, 128, 256, 24, 64, 64])
    c8 = inception_block(c7, [112, 114, 288, 32, 64, 64])
    c9 = inception_block(c8, [256, 160, 320, 32, 128, 128])
    p4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding="same")(c9)
    c10 = inception_block(p4, [256, 160, 320, 32, 128, 128])
    c11 = inception_block(c10, [384, 192, 384, 48, 128, 128])
# flat
    p6 = K.layers.AveragePooling2D(pool_size=(7, 7))(c11)
    drop = K.layers.Dropout(0.4)(p6)
    return K.models.Model(inputs=X,
                          outputs=K.layers.Dense(1000,
                                                 activation='softmax')(drop))
