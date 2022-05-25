#!/usr/bin/env python3
"""flipping module"""

import tensorflow as tf


def flip_image(image):
    """flipping function"""
    return tf.image.flip_left_right(image)
