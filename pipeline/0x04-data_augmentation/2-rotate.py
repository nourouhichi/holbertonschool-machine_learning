#!/usr/bin/env python3
"""rotation module"""

import tensorflow as tf


def rotate_image(image):
    """rotation function"""
    return tf.image.rot90(image)
