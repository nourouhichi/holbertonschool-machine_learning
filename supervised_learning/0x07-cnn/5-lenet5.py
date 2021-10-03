#!/usr/bin/env python3
""" Convolutional Neural Networks """


import tensorflow.keras as K


def lenet5(X):
    """lenet5 with keras"""
    initializer = K.initializers.he_normal()
    c1 = K.layers.Conv2D(6, (5, 5),
                         padding='same', kernel_initializer=initializer,
                         activation='relu')(X)
    p1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)
    c2 = K.layers.Conv2D(16, (5, 5),
                         padding='valid', kernel_initializer=initializer,
                         activation='relu')(p1)
    p2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)
    l1 = K.layers.Dense(120, kernel_initializer=initializer,
                        activation='relu')(K.layers.Flatten()(p2))
    l2 = K.layers.Dense(84, kernel_initializer=initializer,
                        activation='relu')(l1)
    output = K.layers.Dense(10, kernel_initializer=initializer,
                            activation='softmax')(l2)
    model = K.models.Model(inputs=X, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
