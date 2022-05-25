#!/usr/bin/env python3
"""shearing  module"""

import tensorflow as tf


def shear_image(image, intensity):
    """shearing function"""
    return tf.keras.preprocessing.image.random_shear(image.numpy(),
                                                     intensity,
                                                     channel_axis=2)
