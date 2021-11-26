#!/usr/bin/env python3
""" variational autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """variational autoencoder"""

    encoder_input = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0],
        activation='relu')(encoder_input)
    for i in hidden_layers[1::]:
        encoded = keras.layers.Dense(i, activation='relu')(encoded)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)
    epsilon = keras.backend.random_normal(
        shape=(latent_dims,), mean=0.0, stddev=1.0)
    ech = z_mean + keras.backend.exp(sigma / 2) * epsilon
    z = keras.layers.Lambda(ech, output_shape=(
        latent_dims,))([z_mean, sigma])
    encoder = keras.Model(encoder_input, [z_mean, sigma, z])
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(
                                 hidden_layers[-1],
                                 activation='relu')(decoder_input)
    for i in hidden_layers[-2::-1]:
        decoded = keras.layers.Dense(i, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, decoded)
    output = decoder(encoder(encoder_input)[2])
    auto = keras.Model(encoder_input, output)
    rec_loss = keras.losses.binary_crossentropy(
                                                encoder_input,
                                                output)
    rec_loss *= input_dims
    kloss = 1 + sigma - keras.backend.square(z_mean) \
        - keras.backend.exp(sigma)
    kloss = keras.backend.sum(kloss, axis=-1)
    kloss *= -0.5
    vae_loss = keras.backend.mean(rec_loss + kloss)
    auto.compile(optimizer='adam', loss=vae_loss)
    return encoder, decoder, auto
