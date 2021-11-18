#!/usr/bin/env python3
"""bayes optim """
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """bayes omptim"""

    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True):
        """ Init function"""
        l, h = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(l, h, ac_samples).reshape((-1, 1))
        self.X_s = (np.sort(self.X_s)).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
