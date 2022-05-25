#!/usr/bin/env python3
"""monte carlo module"""
import numpy as np


def monte_carlo(env, V, policy,
                episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """monte carlo function"""
    for i in range(episodes):
        res = env.reset()
        episodes_list = []
        for _ in range(max_steps):
            action = policy(res)
            new_res, reward, done, _ = env.step(action)
            episodes_list.append([res, action, reward, new_res])
            if done:
                break
            res = new_res
        episode = np.array(episodes_list, dtype=int)
        mc = 0
        for _, step in enumerate(episode[::-1]):
            res, action, reward, _ = step
            mc = gamma * mc + reward
            if res not in episode[:i, 0]:
                V[res] = V[res] + alpha * (mc - V[res])
    return V
