#!/usr/bin/env python3
""" Convolutional Neural Networks """


import tensorflow as tf


def lenet5(x, y):
    """cloning lenet5"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN")
    c1 = tf.layers.Conv2D(6, (5, 5), padding="same",
                          activation='relu', kernel_initializer=init)(x)
    p1 = tf.layers.max_pooling2d(c1, pool_size=(2, 2), strides=(2, 2))
    c2 = tf.layers.Conv2D(16, (5, 5), padding="valid",
                          activation='relu', kernel_initializer=init)(p1)
    p2 = tf.layers.max_pooling2d(c2, pool_size=(2, 2), strides=(2, 2))
    fc1 = tf.layers.Dense(120, activation="relu", kernel_initializer=init)(tf.layers.Flatten()(p2))
    fc2 = tf.layers.Dense(84, activation="relu", kernel_initializer=init)(fc1)
    y_pred = tf.layers.Dense(10, activation="softmax", kernel_initializer=init)(fc2)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(
        tf.math.argmax(y, axis=1),
        tf.math.argmax(y_pred, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return y_pred, train_op, loss, acc

