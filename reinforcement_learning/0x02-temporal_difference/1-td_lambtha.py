#!/usr/bin/env python3
"""rd talmbtha module"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """td lambtha function"""
    states = V.shape[0]
    el = np.zeros(states)
    for _ in range(episodes):
        res = env.reset()
        for _ in range(max_steps):
            action = policy(res)
            new_res, reward, done, _ = env.step(action)
            el[res] += 1.0
            delta = reward + gamma * V[new_res] - V[res]
            V += alpha * delta * el
            el *= lambtha * gamma
            if done:
                break
            else:
                res = new_res
        return V
