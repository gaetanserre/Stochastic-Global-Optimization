#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from scipy.spatial.distance import pdist, squareform
from .__optimizer__ import Optimizer
from .N_CMA_ES import CMA_ES


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

    return grad


def rbf(x, h=-1):
    sq_dist = pdist(x)
    pairwise_dists = squareform(sq_dist) ** 2
    if h < 0:  # if h < 0, using median trick
        h = np.median(pairwise_dists) + 1e-10
        h = np.sqrt(0.5 * h / np.log(x.shape[0] + 1))

    # compute the rbf kernel
    Kxy = np.exp(-pairwise_dists / h**2 / 2)

    dxkxy = (x * Kxy.sum(axis=1).reshape(-1, 1) - Kxy @ x).reshape(
        x.shape[0], x.shape[1]
    ) / (h**2)

    return Kxy, dxkxy


def svgd(x, logprob_grad, kernel):
    Kxy, dxkxy = kernel(x)

    svgd_grad = (Kxy @ logprob_grad + dxkxy) / x.shape[0]
    return svgd_grad


class NMDS_CMA(Optimizer):
    def __init__(self, domain, n_particles, k_iter, svgd_iter, cma_iter, lr=0.5):
        self.domain = domain
        self.n_particles = n_particles
        self.k_iter = k_iter
        self.svgd_iter = svgd_iter
        self.cma_iter = cma_iter
        self.lr = lr

    def initialize_particles(self, function):
        dim = self.domain.shape[0]

        m_0 = np.random.uniform(self.domain[:, 0], self.domain[:, 1])
        cma = CMA_ES(self.domain, m_0, self.cma_iter)

        uniform_particles = self.n_particles // 2
        cma_particles = self.n_particles - uniform_particles
        mean, std = cma.optimize_stats(function)

        x_cma = np.random.normal(mean, std, size=(cma_particles, dim))
        x_uniform = np.random.uniform(
            self.domain[:, 0], self.domain[:, 1], size=(uniform_particles, dim)
        )
        return np.concatenate((x_cma, x_uniform), axis=0)

    def optimize(self, function, verbose=False):
        logprob_grad = lambda k: (lambda x: -k * gradient(function, x))

        kernel = rbf

        dim = self.domain.shape[0]

        x = self.initialize_particles(function)

        all_points = [x.copy()]
        for k in self.k_iter:
            optimizer = Adam(lr=self.lr)
            for i in range(self.svgd_iter):
                svgd_grad = svgd(x, np.array([logprob_grad(k)(xi) for xi in x]), kernel)
                x = optimizer.step(svgd_grad, x)

                # clamp to domain
                x = np.clip(x, self.domain[:, 0], self.domain[:, 1])

                # save all points
                all_points.append(x.copy())

        evals = np.array([function(xi) for xi in x]).flatten()
        best_idx = np.argmin(evals)
        min_eval = evals[best_idx]
        best_particle = x[best_idx]
        if verbose:
            print(f"Best particle found: {best_particle}. Eval at f(best): {min_eval}.")

        all_points = np.array(all_points).reshape(-1, dim)
        all_evals = np.array([function(xi) for xi in all_points]).flatten()
        return (best_particle, min_eval), all_points, all_evals
