#!/usr/bin/env python3
"""new module"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ forword function"""
    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i],
                            activation=activations[i])
    return prev
