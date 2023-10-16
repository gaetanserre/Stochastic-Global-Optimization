#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from scipy.spatial.distance import pdist, squareform
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

    def update_states(self, mask):
        self.state_m = self.state_m[mask]
        self.state_v = self.state_v[mask]


def gradient(f, x, eps=1e-12):
    f_x = f(x)

    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_p = x.copy()
        x_p[i] += eps
        grad[i] = (f(x_p) - f_x) / eps

    # We remove the number of evaluations required to estimate the gradient
    # f.n -= x.shape[0]

    return grad, f_x


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


class NMDS_particles(Optimizer):
    def __init__(
        self,
        domain,
        n_particles,
        k_iter,
        svgd_iter,
        distance_q=0.5,
        value_q=0.3,
        lr=0.2,
    ):
        self.domain = domain
        self.n_particles = n_particles
        self.k_iter = k_iter
        self.svgd_iter = svgd_iter
        self.distance_q = distance_q
        self.value_q = value_q
        self.lr = lr

    def remove_particles(self, x, x_new, x_values):
        if x_new.shape[0] > 10:
            distance = np.linalg.norm(x - x_new, axis=1)
            dist_quantile = np.quantile(distance, q=self.distance_q)
            dist_mask = distance < dist_quantile

            value_quantile = np.quantile(x_values, q=self.value_q)
            value_mask = x_values > value_quantile

            mask = ~(dist_mask & value_mask)

            if np.all(mask):
                return x_new, np.ones(x_new.shape[0], dtype=bool)
            else:
                return x_new[mask], mask
        else:
            return x_new, np.ones(x_new.shape[0], dtype=bool)

    def optimize(self, function, verbose=False):
        kernel = rbf

        dim = self.domain.shape[0]

        x = np.random.uniform(
            self.domain[:, 0], self.domain[:, 1], size=(self.n_particles, dim)
        )

        n_particles = self.n_particles

        all_points = [x.copy()]
        for k in self.k_iter:
            optimizer = Adam(lr=self.lr)
            for i in range(self.svgd_iter):
                logprob_grad_array = [np.zeros(dim)] * n_particles
                f_evals = [0] * n_particles

                for j in range(n_particles):
                    grad, f_eval = gradient(function, x[j])
                    logprob_grad_array[j] = -k * grad
                    f_evals[j] = f_eval

                svgd_grad = svgd(x, logprob_grad_array, kernel)
                # x_new = x + 1e-8 * svgd_grad
                x_new = optimizer.step(svgd_grad, x)

                # clamp to domain
                x_new = np.clip(x_new, self.domain[:, 0], self.domain[:, 1])

                x_new, mask = self.remove_particles(x, x_new, f_evals)
                n_particles = x_new.shape[0]
                optimizer.update_states(mask)

                x = x_new

                # save all points
                all_points.append(x.copy())

        evals = np.array([function(xi) for xi in x]).flatten()
        best_idx = np.argmin(evals)
        min_eval = evals[best_idx]
        best_particle = x[best_idx]
        if verbose:
            print(f"Best particle found: {best_particle}. Eval at f(best): {min_eval}.")

        np_all_points = None
        for i, np_seq in enumerate(all_points):
            if i == 0:
                np_all_points = np_seq
            else:
                np_all_points = np.concatenate((np_all_points, np_seq), axis=0)

        all_evals = np.array([function(xi) for xi in np_all_points]).flatten()
        return (best_particle, min_eval), np_all_points, all_evals
