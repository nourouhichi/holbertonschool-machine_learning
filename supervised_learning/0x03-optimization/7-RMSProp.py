#!/usr/bin/env python3
"""RMSProp"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """rmsprop"""
    sd = beta2 * s + (1 - beta2) * (grad**2)
    w = var - alpha * grad / (sd ** (1/2) + epsilon)
    return w, sd