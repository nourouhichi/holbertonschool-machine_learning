#!/usr/bin/env python3
"""hmm backword"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ hmm backword """
    T, = Observation.shape
    N, _ = Emission.shape
    b = np.zeros((N, T))
    b[:, T - 1] = np.ones(N)
    for i in range(T - 2, -1, -1):
        for j in range(N):
            b[j, i] = (b[:, i + 1] * Emission[
                         :, Observation[i + 1]]).dot(Transition[j, :])
    P = np.sum(b[:, 0] * Emission[:, Observation[0]] * Initial.T)
    return P, b
