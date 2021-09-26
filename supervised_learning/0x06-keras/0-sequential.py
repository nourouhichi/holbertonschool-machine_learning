#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """building a neural network with keras"""
    model = K.Sequential()
    model.add(
             K.layers.Dense(layers[0], activation=activations[0],
                            input_dim=nx,
                            kernel_regularizer=K.regularizers.l2(
                           lambtha)))
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(
                                lambtha)))
    return model
