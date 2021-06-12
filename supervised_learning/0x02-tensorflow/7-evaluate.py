#!/usr/bin/env python3
"""evaluation"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """return evaluation of the neural network"""

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(save_path + ".meta")
        new_saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection("accuracy")[0]
        y_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        cost = sess.run(loss, feed_dict={x: X, y: Y})
        acc = sess.run(accuracy, feed_dict={x: X, y: Y})
        return y_pred, acc, cost
