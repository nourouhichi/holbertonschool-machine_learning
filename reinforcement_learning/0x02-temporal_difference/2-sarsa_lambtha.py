#!/usr/bin/env python3
"""sarsa lambtha module"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """darsa lambtha function"""
    init_epsilon = epsilon
    el = np.zeros(Q.shape)
    for i in range(episodes):
        res = env.reset()
        action = 0
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[res, :])
        for _ in range(max_steps):
            res_new, reward, done, _ = env.step(action)
            greedy = 0
            if np.random.uniform(0, 1) < epsilon:
                greedy = env.action_space.sample()
            else:
                greedy = np.argmax(Q[res, :])
            el *= gamma * epsilon
            el[res, action] += (1.0)
            delta = reward + gamma * Q[res_new, greedy] - Q[res, action]
            Q += alpha * delta * el
            if done:
                break
            res = res_new
        if epsilon < min_epsilon:
            epsilon = min_epsilon
        else:
            epsilon *= init_epsilon * np.exp(-epsilon_decay * i)
    return Q
