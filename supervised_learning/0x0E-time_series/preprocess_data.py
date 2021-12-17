#!/usr/bin/env python3
import pandas as pd
import tensorflow as tf
import numpy as np


class WindowGenerator:
    """windowing data"""
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(
            self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(
            self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[
                    name]] for name in self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        """timeseries data making for rnn use"""
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False)

        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)


def preprocess():
    # converting to datetime
    df = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")
    df.drop(['Volume_(Currency)', "Weighted_Price", "High",
             "Low", "Volume_(BTC)", "Open"], axis=1, inplace=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
    df.dropna(inplace=True)
    df = df[df["Timestamp"] >= "2017-01-01 00:00:00"]
    # windowing per hour
    df = df.set_index("Timestamp").asfreq("1H")
    df.dropna(inplace=True)
    # train, val, test slicing
    i = df.shape[0]
    train_set = df[0:int(i*0.7)]
    validation_set = df[int(i*0.7):int(i*0.9)]
    test_set = df[int(i*0.9):]
    # normalization
    train_set.dropna(inplace=True)
    train_mean = train_set.mean()
    train_std = train_set.std()
    train_set = (train_set - train_mean) / train_std

    test_mean = test_set.mean()
    test_std = test_set.std()
    test_set = (test_set - test_mean) / test_std

    validation_mean = validation_set.mean()
    validation_std = validation_set.std()
    validation_set = (validation_set - validation_mean) / validation_std

    window = WindowGenerator(input_width=24, label_width=1, shift=1,
                             label_columns=["Close"],
                             train_df=train_set, val_df=validation_set,
                             test_df=test_set)
    return(window)
