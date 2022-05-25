#!/usr/bin/env python3
"""hue module"""

import tensorflow as tf


def change_hue(image, delta):
    """hue function"""
    return tf.image.adjust_hue(image, delta)
