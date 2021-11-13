#!/usr/bin/env python3
"""markov chain"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """"forword  markov chain"""

    try:
        T, = Observation.shape
        N = Emission.shape[0]
        alpha = np.zeros((N, T))
        alpha[:, 0] = Initial.T * Emission[:, Observation[0]]
        for i in range(1, T):
            for j in range(N):
                alpha[j, i] = alpha[:, i - 1].dot(
                    Transition[:, j]) * Emission[j, Observation[i]]
        P = np.sum(alpha[:, T - 1])
        return P, alpha
    except Exception:
        return None, None
