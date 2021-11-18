#!/usr/bin/env python3
"""bayes optim """
from scipy.stats import norm
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
        min, h = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(min, h, ac_samples).reshape((-1, 1))
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """next best sample location"""
        mu, cov = self.gp.predict(self.X_s)
        z = np.zeros(cov.shape[0])
        if self.minimize:
            mu_ = np.min(self.gp.Y)
            ip = mu_ - mu - self.xsi
        else:
            mu_s_opt = np.max(self.gp.Y)
            ip = mu - mu_s_opt - self.xsi
        for i in range(cov.shape[0]):
            if cov[i] > 0:
                z[i] = ip[i] / cov[i]
            else:
                z[i] = 0
            EI = ip * norm.cdf(z) + cov * norm.pdf(z)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
