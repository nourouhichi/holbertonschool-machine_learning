#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """training model using keras"""
    if early_stopping:
        callback = K.callbacks.EarlyStopping(patience=patience)
    return network.fit(data, labels, epochs=epochs, callbacks=[callback],
                       batch_size=batch_size, verbose=verbose,
                       validation_data=validation_data,
                       shuffle=shuffle)
