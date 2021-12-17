#!/usr/bin/env python3

from preprocess_data import preprocess
import tensorflow as tf


MAX_EPOCHS = 20


def compile_and_fit(model, window, patience=2):
    """training function"""
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val)
    return history


window = preprocess()
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(model, window)
val = model.evaluate(window.val)
test = model.evaluate(window.test, verbose=0)
model.save('lstm.h5')
