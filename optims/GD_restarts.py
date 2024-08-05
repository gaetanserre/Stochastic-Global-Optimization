#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .__optimizer__ import Optimizer


class Adam:
    def __init__(
        self,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.state_m = 0
        self.state_v = 0
        self.state_v_max = 0
        self.t = 0

    def step(self, grad, params):
        self.t += 1

        grad = -grad

        self.state_m = self.betas[0] * self.state_m + (1 - self.betas[0]) * grad
        self.state_v = self.betas[1] * self.state_v + (1 - self.betas[1]) * grad**2

        m_hat = self.state_m / (1 - self.betas[0] ** self.t)
        v_hat = self.state_v / (1 - self.betas[1] ** self.t)

        if self.amsgrad:
            self.state_v_max = np.maximum(self.state_v_max, v_hat)
            return self.lr * m_hat / (np.sqrt(self.state_v_max) + self.eps)
        else:
            return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def gradient(f, x, eps=1e-12):
    f_x = f(x)

    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_p = x.copy()
        x_p[i] += eps
        grad[i] = (f(x_p) - f_x) / eps

    return grad, f_x


def restart(x, domain):
    return np.random.uniform(domain[:, 0], domain[:, 1], size=x.shape)


class GD_restarts(Optimizer):
    def __init__(self, domain, n_particles, iter, lr=0.5):
        self.domain = domain
        self.n_particles = n_particles
        self.iter = iter
        self.lr = lr

    def optimize(self, function, verbose=False):

        dim = self.domain.shape[0]

        x = np.random.uniform(
            self.domain[:, 0], self.domain[:, 1], size=(self.n_particles, dim)
        )

        all_points = [x.copy()]
        all_evals = []
        optimizer = Adam(lr=self.lr)
        steps = np.ones(self.n_particles)
        for i in range(self.iter):
            grads = [0] * self.n_particles
            fs = [0] * self.n_particles
            for j, xj in enumerate(x):
                grad, f_xj = gradient(function, xj)
                grads[j] = -grad
                fs[j] = f_xj
            all_evals.append(fs)
            if i >= 1:
                for j, xj in enumerate(x):
                    if fs[j] > all_evals[-2][j]:
                        x[j] = restart(xj, self.domain)
                        grads[j] = np.zeros_like(grads[j])
                        steps[j] = 1
            x += (
                (self.lr / steps).reshape(1, -1) * np.array(grads).T
            ).T  # optimizer.step(np.array(grads), x)
            steps += 1

            # clamp to domain
            x = np.clip(x, self.domain[:, 0], self.domain[:, 1])

            # save all points
            all_points.append(x.copy())

        all_points = np.array(all_points).reshape(-1, dim)
        all_evals = np.array(all_evals).flatten()
        best_idx = np.argmin(all_evals)
        min_eval = all_evals[best_idx]
        best_particle = all_points[best_idx]
        if verbose:
            print(f"Best particle found: {best_particle}. Eval at f(best): {min_eval}.")

        return (best_particle, min_eval), all_points, all_evals
