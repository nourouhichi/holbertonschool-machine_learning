#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """testing the model"""
    return network.evaluate(data, labels, verbose=verbose)
