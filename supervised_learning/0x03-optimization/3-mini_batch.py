#!/usr/bin/env python3
"""new module"""

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
        if m % batch_size == 0:
            n_batches = m // batch_size
        else:
            n_batches = (m // batch_size) + 1
        for epoch in range(epochs + 1):
            cost_t = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accu_t = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accu_val = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(accu_t))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accu_val))
            if epoch < epochs:
                X_shuf, Y_shuf = shuffle_data(X_train, Y_train)
                for i in range(n_batches):
                    start = i * batch_size
                    if i == n_batches - 1 and  m % batch_size != 0:
                        limit = m
                    else:
                        limit = start + batch_size
                    batched_x = X_shuf[start:limit]
                    batched_y = Y_shuf[start:limit]
                    sess.run(
                        train_op,
                        feed_dict={x: batched_x,
                                   y: batched_y})
                    if (i) % 100 == 0 and i is not 0:
                        tr_cost = sess.run(
                            loss,
                            feed_dict={x: batched_x,
                                       y: batched_y})
                        tr_acc = sess.run(
                            accuracy,
                            feed_dict={x: batched_x,
                                       y: batched_y})
                        print("\t\tStep {}:".format(i))
                        print("\t\tCost: {}".format(tr_cost))
                        print("\t\tAccuracy: {}".format(tr_acc))
        return new_saver.save(sess, save_path)
