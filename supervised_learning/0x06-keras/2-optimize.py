#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """adam optm"""
    opt = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(opt,
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])
