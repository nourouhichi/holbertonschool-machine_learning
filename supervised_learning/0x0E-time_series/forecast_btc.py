#!/usr/bin/env python3

from re import T
from pandas.core.indexes.base import Index
from preprocess_data import preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


MAX_EPOCHS = 10


def compile_and_fit(model, window):
    """training function"""
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val)
    return history


window = preprocess()
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(24, return_sequences=False),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(model, window)
val = model.evaluate(window.val)
pred = model.predict(window.test, verbose=0)
df = window.test_df
pred = pd.DataFrame(pred, index = df.index[24:])
plt.plot(pred, label='Prediction')
plt.plot(df["Close"], label='Label')
plt.xlabel("Time")
plt.ylabel("Close")
plt.legend()
plt.show()
model.summary()
model.save('lstm.h5')
