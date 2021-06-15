#!/usr/bin/env python3
"""new module"""

import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
                    X_train, Y_train,
                    X_valid, Y_valid,
                    batch_size=32,
                    epochs=5,
                    load_path="/tmp/model.ckpt",
                    save_path="/tmp/model.ckpt"):
    """training function"""
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(load_path + ".meta")
        new_saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection("accuracy")[0]
        train_op = tf.get_collection("train_op")[0]
        m = X_train.shape[0]

        for epoch in range(epochs + 1):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            cost_t = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accu_t = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accu_val = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(accu_t))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accu_val))
            Xt_batched = np.array_split(X_train, (m / batch_size))
            Yt_batched = np.array_split(Y_train, (m / batch_size))
            if m % batch_size == 0:
                n_batches = m / batch_size
            else:
                n_batches = (m / batch_size) + 1
            if epoch < epochs:
                for i in range(n_batches - 1):
                    tr_cost = sess.run(
                        loss,
                        feed_dict={x: Xt_batched[i], y: Yt_batched[i]})
                    tr_acc = sess.run(
                        accuracy,
                        feed_dict={x: Xt_batched[i], y: Yt_batched[i]})
                    if i % 100 == 0 and i is not 0:
                        print("\t\tStep {}:".format(i))
                        print("\t\tCost: {}".format(tr_cost))
                        print("\t\tAccuracy: {}".format(tr_acc))
                    sess.run(
                        train_op,
                        feed_dict={x: Xt_batched[i], y: Yt_batched[i]})
        return new_saver.save(sess, save_path)
