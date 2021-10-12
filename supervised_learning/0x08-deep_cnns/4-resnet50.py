#!/usr/bin/env python3
""" Resnet-50 module"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ResNet-50 architecture """
    init = K.initializers.he_normal()
    input = K.Input((224, 224, 3))
    c1 = K.layers.Conv2D(64, (7, 7), padding='same', strides=2,
                         kernel_initializer=init)(input)
    norm1 = K.layers.BatchNormalization()(c1)
    l1 = K.layers.Activation('relu')(norm1)
    p1 = K.layers.MaxPooling2D((3, 3),
                               strides=(2, 2),
                               padding='same')(l1)

    proj = projection_block(p1, [64, 64, 256], 1)
    id = identity_block(proj, [64, 64, 256])
    id = identity_block(id, [64, 64, 256])
    proj = projection_block(id, [128, 128, 512], 2)
    id = identity_block(proj, [128, 128, 512])
    id = identity_block(id, [128, 128, 512])
    id = identity_block(id, [128, 128, 512])
    proj = projection_block(id, [256, 256, 1024], 2)
    id = identity_block(proj, [256, 256, 1024])
    id = identity_block(id, [256, 256, 1024])
    id = identity_block(id, [256, 256, 1024])
    id = identity_block(id, [256, 256, 1024])
    id = identity_block(id, [256, 256, 1024])
    proj = projection_block(id, [512, 512, 2048], 2)
    id = identity_block(proj, [512, 512, 2048])
    id = identity_block(id, [512, 512, 2048])
    p2 = K.layers.AveragePooling2D((7, 7))(id)
    output = K.layers.Dense(
                            1000,
                            activation='softmax',
                            kernel_initializer=init)(p2)
    return K.models.Model(inputs=input, outputs=output)
