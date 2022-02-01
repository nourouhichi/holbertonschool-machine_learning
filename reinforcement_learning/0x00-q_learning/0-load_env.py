#!/usr/bin/env python3
""" reinforcement learning """
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Qlearning with gym"""
    return gym.make("FrozenLake-v0",
                    is_slippery=is_slippery,
                    desc=desc, map_name=map_name)
