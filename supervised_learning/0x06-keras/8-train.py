#!/usr/bin/env python3
"""new module"""


import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """training model using keras"""
    def scheduler(epoch):
        return alpha / (1 + decay_rate * epoch)
    calls = []
    if learning_rate_decay:
        callback_1 = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        calls.append(callback_1)
    if save_best:
        callback_2 = K.callbacks.ModelCheckpoint(filepath, monitor="val_loss",
                                                 save_best_only=True)
        calls.append(callback_2)
    if early_stopping:
        callback = K.callbacks.EarlyStopping(patience=patience)
        calls.append(callback)
    return network.fit(data, labels, epochs=epochs,
                       callbacks=calls,
                       batch_size=batch_size, verbose=verbose,
                       validation_data=validation_data,
                       shuffle=shuffle)
