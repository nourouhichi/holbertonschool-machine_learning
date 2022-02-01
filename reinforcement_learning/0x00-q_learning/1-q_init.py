#!/usr/bin/env python3
""" reinforcement learning """
import numpy as np


def q_init(env):
    """ Initialize the Qtable """
    return np.zeros([env.observation_space.n, env.action_space.n])
