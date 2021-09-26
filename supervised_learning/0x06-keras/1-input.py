#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """model building without using seq"""
    inputs = K.Input(shape=(nx,))
    dense = K.layers.Dense(layers[0], activation=activations[0],
                           kernel_regularizer=K.regularizers.l2(
                           lambtha))
    outputs = dense(inputs)
    for i in range(1, len(layers)):
        outputs = K.layers.Dropout(1 - keep_prob)(outputs)
        outputs = K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(
                                lambtha))(outputs)

    return K.Model(inputs=inputs, outputs=outputs)
