#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """convert to one hot"""
    return K.utils.to_categorical(labels, classes)
