#!/usr/bin/env python3
"""sparce network"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ sparce autoencoder"""
    input_en = keras.Input(shape=(input_dims,))
    enc = keras.layers.Dense(
                             hidden_layers[0],
                             activation='relu')(input_en)
    for i in hidden_layers[1::]:
        enc = keras.layers.Dense(i, activation='relu')(enc)
    lt = keras.layers.Dense(
                            latent_dims,
                            activation='relu',
                            activity_regularizer=keras.regularizers.l1(
                                lambtha))(enc)
    encoder = keras.Model(input_en, lt)
    input_dec = keras.Input(shape=(latent_dims,))
    dec = keras.layers.Dense(
        hidden_layers[-1], activation='relu')(input_dec)
    for i in hidden_layers[-2::-1]:
        dec = keras.layers.Dense(i, activation='relu')(dec)
    dec = keras.layers.Dense(input_dims, activation='sigmoid')(dec)
    decoder = keras.Model(input_dec, dec)
    autoenc = keras.Model(input_en, decoder(encoder(input_en)))
    autoenc.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, autoenc
