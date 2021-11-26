#!/usr/bin/env python3
""" variational autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """variational autoencoder"""

    input_en = keras.Input(shape=(input_dims,))
    enc = keras.layers.Dense(
                             hidden_layers[0],
                             activation='relu')(input_en)
    for i in hidden_layers[1::]:
        enc = keras.layers.Dense(i, activation='relu')(enc)

    mean = keras.layers.Dense(latent_dims, activation=None)(enc)
    sigma = keras.layers.Dense(latent_dims, activation=None)(enc)
    ep = keras.backend.random_normal(
                                     shape=(latent_dims,),
                                     mean=0.0, stddev=1.0)
    ech = mean + keras.backend.exp(sigma / 2) * ep
    z = keras.layers.Lambda(ech, output_shape=(
        latent_dims,))([mean, sigma])

    encoder = keras.Model(input_en, [mean, sigma, z])
    input_dec = keras.Input(shape=(latent_dims,))
    dec = keras.layers.Dense(
                             hidden_layers[-1],
                             activation='relu')(input_dec)
    for i in hidden_layers[-2::-1]:
        dec = keras.layers.Dense(i, activation='relu')(dec)
    dec = keras.layers.Dense(input_dims, activation='sigmoid')(dec)
    decoder = keras.Model(input_dec, dec)
    output = decoder(encoder(input_en)[2])
    autoen = keras.Model(input_en, output)

    rec_l = keras.losses.binary_crossentropy(
        input_en, output)
    rec_l *= input_dims
    loss = 1 + sigma - keras.backend.square(mean) \
        - keras.backend.exp(sigma)
    loss = keras.backend.sum(loss, axis=-1)
    loss *= -0.5
    vae_l = keras.backend.mean(rec_l + loss)

    autoen.compile(optimizer='adam', loss=vae_l)

    return encoder, decoder, autoen
