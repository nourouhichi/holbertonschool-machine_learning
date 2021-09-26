#!/usr/bin/env python3
"""new module"""


from re import X
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """model building without using seq"""
    inputs = K.Input(shape=(nx,))
    outputs = inputs

    for i in range(len(layers)):
        if i != 0:
            outputs = K.layers.Dropout(1 - keep_prob)(outputs)
        outputs = K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(
                                lambtha))(outputs)

    return K.Model(inputs=inputs, outputs=outputs)
