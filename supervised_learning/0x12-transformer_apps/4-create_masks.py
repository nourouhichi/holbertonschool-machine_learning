#!/usr/bin/env python3
""" creates all masks"""
import tensorflow as tf


def padding(x):
    """creates a padding mask"""
    cast = tf.cast(tf.math.equal(x, 0), tf.float32)
    return cast[:, tf.newaxis, tf.newaxis, :]


def create_masks(inputs, target):
    """creates all masks for training/validation"""
    encoder_mask = padding(inputs)
    decoder_mask = padding(inputs)
    batch_size, len_out = target.shape
    la_mask = 1 - tf.linalg.band_part(tf.ones(shape=(
            batch_size, 1, len_out, len_out)), -1, 0)
    dt_mask = padding(target)
    comb_mask = tf.maximum(dt_mask, la_mask)
    return encoder_mask, comb_mask, decoder_mask
