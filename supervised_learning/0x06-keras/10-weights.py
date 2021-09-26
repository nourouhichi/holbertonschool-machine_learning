#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saving weights"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """loading weights"""
    network.load_weights(filename)
