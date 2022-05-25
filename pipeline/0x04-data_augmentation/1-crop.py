#!/usr/bin/env python3
"""cropping module"""
import tensorflow as tf


def crop_image(image, size):
    """cropping function"""
    return tf.image.random_crop(image, size)
