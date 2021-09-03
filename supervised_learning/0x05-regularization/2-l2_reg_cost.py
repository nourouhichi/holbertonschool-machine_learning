#!/usr/bin/env python3
""" Reg """

import tensorflow as tf


def l2_reg_cost(cost):
    """cost function reg"""
    return cost + tf.contrib.l2_regularizer()
