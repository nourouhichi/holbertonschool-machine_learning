#!/usr/bin/env python3
""" Convolutional Neural Networks """


import tensorflow as tf


def lenet5(x, y):
    """"lenet4 cloning"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_IN")
    c1 = tf.nn.relu(tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        kernel_initializer=initializer,
        name="c1",
        )(x))
    p1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)
    c2 = tf.nn.relu(tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        kernel_initializer=initializer,
        name="c2",
        )(p1))
    p2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)
    l1 = tf.nn.relu(tf.layers.Dense(120,
                                    kernel_initializer=initializer)
                    (tf.layers.Flatten()(p2)))
    l2 = tf.nn.relu(tf.layers.Dense(80,
                                    kernel_initializer=initializer)(l1))
    lf = tf.layers.Dense(10,
                         kernel_initializer=initializer,)(l2)
    y_pred = tf.nn.softmax(lf)
    loss = tf.losses.softmax_cross_entropy(y, lf)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(
        tf.math.argmax(y, axis=1),
        tf.math.argmax(lf, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train_op, loss, acc
