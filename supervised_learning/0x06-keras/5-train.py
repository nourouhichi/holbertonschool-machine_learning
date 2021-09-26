#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                verbose=True, shuffle=False):
    """training model using keras"""
    return network.fit(data, labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       validation_data=validation_data,
                       shuffle=shuffle)
