#!/usr/bin/env python3
"""brightness module"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """change brightness """
    return tf.image.adjust_brightness(image, max_delta)
