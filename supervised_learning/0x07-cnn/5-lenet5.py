#!/usr/bin/env python3
""" Convolutional Neural Networks """


import tensorflow.keras as K


def lenet5(X):
    """lenet5 with keras"""
    m, x, y, z = X.shape
    init = K.initializers.he_normal()
    c1 = K.layers.Conv2D(6, (5, 5), padding="same",
                         activation="relu",
                         kernel_initializer=init)(X)
    p1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)
    c2 = K.layers.Conv2D(16, (5, 5), padding="valid",
                         activation="relu", kernel_initializer=init)(p1)
    p2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)
    fc1 = K.layers.Dense(120, activation="relu",
                         kernel_initializer=init)(K.layers.Flatten()(p2))
    fc2 = K.layers.Dense(84, activation="relu", kernel_initializer=init)(fc1)
    y_pred = K.layers.Dense(10, kernel_initializer=init,
                            activation="softmax")(fc2)
    model = K.Model(inputs=X, outputs=y_pred)
    model.compile(optimizer="Adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
