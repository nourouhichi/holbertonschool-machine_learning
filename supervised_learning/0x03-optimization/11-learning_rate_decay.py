#!/usr/bin/env python3
"""Normalization"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ training"""
    return alpha / (1 + (decay_rate * np.floor(global_step / decay_step)))
