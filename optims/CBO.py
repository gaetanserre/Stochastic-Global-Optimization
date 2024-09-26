#
# Created in 2024 by Gaëtan Serré
#

from .__optimizer__ import Optimizer

import numpy as np
from scipy import special
from numpy.random import normal


class CBO(Optimizer):
    def __init__(
        self,
        domain,
        n_iter,
        n_particles,
        dt=0.01,
        lamda=1.0,
        alpha=1.0,
        sigma=1.0,
        eps=1e-3,
    ):
        self.domain = domain
        self.n_iter = n_iter
        self.n_particles = n_particles
        self.dt = dt
        self.lamda = lamda
        self.alpha = alpha
        self.sigma = sigma
        self.eps = eps

    def weights(self, x, f):
        weights = np.zeros((len(x), 1))
        for i in range(len(x)):
            weights[i] = np.exp(-self.alpha * f(x[i]))
        return weights

    def consensus(self, x, f):
        weights = self.weights(x, f)
        return np.sum(weights * x, axis=0) / np.sum(weights)

    def drift(self, x, f):
        return x - self.consensus(x, f)

    def heaviside(self, x):
        return 0.5 * special.erf((1 / self.eps) * x) + 0.5

    def noise(self, drift):
        z = normal(0, 1, size=(drift.shape))
        return (
            self.sigma
            * np.sqrt(self.dt)
            * z
            * np.linalg.norm(drift, axis=-1, keepdims=True)
        )

    def optimize(self, function, verbose=False):
        x = np.random.uniform(
            self.domain[:, 0], self.domain[:, 1], (self.n_particles, len(self.domain))
        )

        for _ in range(self.n_iter):
            drift = self.drift(x, function)
            noise = self.noise(drift)
            # print(np.linalg.norm(drift, axis=-1, keepdims=True))
            x -= self.lamda * drift * self.heaviside(x) * self.dt + self.noise(drift)
            np.nan_to_num(x, copy=False, nan=1e8)
            x = np.clip(x, self.domain[:, 0], self.domain[:, 1])

        values = np.array([function(xi) for xi in x])
        best = x[np.argmin(values)]
        return (best, function(best)), x, values
