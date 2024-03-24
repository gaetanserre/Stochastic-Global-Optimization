#
# Created in 2024 by Gaëtan Serré
#

from typing import Tuple
import numpy as np
from .__optimizer__ import Optimizer


def gradient(f, x, eps=1e-12):
    f_x = f(x)

    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_p = x.copy()
        x_p[i] += eps
        grad[i] = (f(x_p) - f_x) / eps

    return grad, f_x


class Langevin(Optimizer):
    def __init__(self, domain, n_iter, kappa, init_lr=0.5):
        self.domain = domain
        self.n_iter = n_iter
        self.kappa = kappa
        self.init_lr = init_lr

    def optimize(self, function, verbose=False):

        x_dim = self.domain.shape[0]
        xi = np.random.normal(0, 1, x_dim)
        samples = []
        values = []
        for i in range(self.n_iter):
            grad, fs = gradient(function, xi)
            grad = self.kappa * grad

            step_size = self.init_lr / (i + 1) ** 2

            xi = (
                xi
                - step_size * grad
                + np.sqrt(2 * step_size) * np.random.normal(0, 1, x_dim)
            )

            xi = np.clip(xi, self.domain[:, 0], self.domain[:, 1])

            samples.append(xi)
            values.append(fs)
        samples = np.array(samples)
        values = np.array(values)
        best_idx = np.argmin(values)
        return (samples[best_idx], values[best_idx]), samples, values
