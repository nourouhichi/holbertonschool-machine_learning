#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def save_model(network, filename):
    """saving a model"""
    network.save(filename)


def load_model(filename):
    """loading a model"""
    return K.models.load_model(filename)
