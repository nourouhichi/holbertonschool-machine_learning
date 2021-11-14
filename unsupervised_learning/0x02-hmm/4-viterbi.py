#!/usr/bin/env python3
"""Viterbi"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Viterbi Algorithm """
    try:
        T, = Observation.shape
        N, _ = Emission.shape
        o = np.zeros((N, T))
        prev = np.zeros((N, T))
        o[:, 0] = Initial.T * Emission[:, Observation[0]]
        for i in range(1, T):
            for j in range(N):
                p = o[:, i - 1] * Emission[
                      j, Observation[i]] * Transition[:, j]
                o[j, i] = np.max(p)
                prev[j, i] = np.argmax(p, 0)
        P = np.max(o[:, T - 1])
        S = []
        current = np.argmax(o[:, T - 1])
        S.append(current)
        for i in range(T - 1, 0, -1):
            current = int(prev[current, i])
            S.append(current)
        return S[::-1], P
    except Exception:
        return None, None
