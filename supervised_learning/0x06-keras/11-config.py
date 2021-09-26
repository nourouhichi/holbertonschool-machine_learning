#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def save_config(network, filename):
    """saving config"""
    with open(filename, "w") as f:
        f.write(network.to_json())


def load_config(filename):
    """loading config"""
    with open(filename, "r") as f:
        reading = f.read()
    return K.models.model_from_json(reading)
