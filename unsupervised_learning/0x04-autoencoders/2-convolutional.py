#!/usr/bin/env python3
"""cnn autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """cnn autoencoder"""
    input_en = keras.Input(shape=input_dims)
    enc = keras.layers.Conv2D(
                              filters[0], (3, 3),
                              activation='relu',
                              padding='same')(input_en)
    enc = keras.layers.MaxPooling2D((2, 2), padding='same')(enc)
    for i in range(1, len(filters)):
        enc = keras.layers.Conv2D(
                                  filters[i],
                                  (3, 3),
                                  activation='relu',
                                  padding='same')(enc)
        enc = keras.layers.MaxPooling2D((2, 2), padding='same')(enc)
    encoder = keras.Model(input_en, enc)
    input_dec = keras.Input(shape=latent_dims)
    dec = keras.layers.Conv2D(
                              filters[-1],
                              (3, 3),
                              activation='relu',
                              padding='same')(input_dec)
    dec = keras.layers.UpSampling2D((2, 2))(dec)
    for i in range(len(filters) - 2, 0, -1):
        dec = keras.layers.Conv2D(
                                  filters[i],
                                  (3, 3), padding='same',
                                  activation='relu')(dec)
        dec = keras.layers.UpSampling2D((2, 2))(dec)
    dec = keras.layers.Conv2D(
                              filters[0],
                              (3, 3), padding='valid',
                              activation='relu')(dec)
    dec = keras.layers.UpSampling2D((2, 2))(dec)
    dec = keras.layers.Conv2D(
                              input_dims[-1],
                              (3, 3), activation='sigmoid',
                              padding='same')(dec)
    decoder = keras.Model(input_dec, dec)
    autoen = keras.Model(input_en, decoder(encoder(input_en)))
    autoen.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, autoen
