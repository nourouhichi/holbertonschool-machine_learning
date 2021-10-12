#!/usr/bin/env python3
""" DenseNet-121 mudule"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """DenseNet-121 architecture """
    init = K.initializers.he_normal()
    input = K.Input(shape=(224, 224, 3))
    norm = K.layers.BatchNormalization(axis=3)(input)
    ly = K.layers.Activation('relu')(norm)
    c = K.layers.Conv2D(2 * growth_rate,
                        (7, 7),
                        (2, 2),
                        kernel_initializer=init,
                        padding='same')(ly)
    p = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=2,
                              padding="same")(c)
    layer, nb_filters = dense_block(p, 2 * growth_rate, growth_rate, 6)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 12)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 24)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 16)
    layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      strides=(1, 1))(layer)
    output = K.layers.Dense(units=1000, activation='softmax',
                            kernel_initializer=init)(layer)
    return K.models.Model(inputs=input, outputs=output)
