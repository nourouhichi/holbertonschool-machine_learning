#!/usr/bin/env python3
"""module"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """predicting"""
    return network.predict(data, verbose=verbose)
