#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from scipy.spatial.distance import pdist, squareform
from .__optimizer__ import Optimizer
from filterpy.monte_carlo import residual_resample


class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def step(self, grad, params):
        pass

    def update_states(self, mask):
        pass


class Adam(Optimizer):
    def __init__(
        self,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    ):
        super().__init__(lr)
        self.init(betas, eps, amsgrad)

    def init(self, betas=(0.9, 0.999), eps=1e-8, amsgrad=False):
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


class Static_Optimizer(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)

    def step(self, grad, params):
        return params + self.lr * grad


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


def rbf(x, sigma=-1):
    sq_dist = pdist(x)
    pairwise_dists = squareform(sq_dist) ** 2
    if sigma < 0:  # if sigma < 0, using median trick
        sigma = np.median(pairwise_dists) + 1e-10
        sigma = np.sqrt(0.5 * sigma / np.log(x.shape[0] + 1))

    # compute the rbf kernel
    Kxy = np.exp(-pairwise_dists / sigma**2 / 2)

    dxkxy = (x * Kxy.sum(axis=1).reshape(-1, 1) - Kxy @ x).reshape(
        x.shape[0], x.shape[1]
    ) / (sigma**2)

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
        distance_q=0.5,  # 0
        value_q=0.3,  # 0
        lr=0.2,
        adam=True,
    ):
        self.domain = domain
        self.n_particles = n_particles
        self.k_iter = k_iter
        self.svgd_iter = svgd_iter
        self.distance_q = distance_q
        self.value_q = value_q
        self.lr = lr
        self.adam = adam

    def move(self, x, logprob_grad_array, kernel, optimizer):
        svgd_grad = svgd(x, logprob_grad_array, kernel)
        x = optimizer.step(svgd_grad, x)

        # clamp to domain
        x = np.clip(x, self.domain[:, 0], self.domain[:, 1])

        return x

    @staticmethod
    def neff(weights):
        return 1.0 / np.sum(weights**2)

    @staticmethod
    def resample_from_indexes(x, weights, indexes, optimizer):
        x[:] = x[indexes]

        rnd_noise = np.random.uniform(-1, 1, x.shape) * 1e-6
        x += rnd_noise

        weights = np.ones(x.shape[0]) / x.shape[0]

        optimizer.init()

        return x, weights

    def resample_from_indexes_no_duplicates(self, x, weights, indexes, optimizer):
        # Remove duplicates
        unique_indexes = np.sort(np.unique(indexes))
        if unique_indexes.shape[0] < 10:
            # print("Not enough unique indexes")
            """if self.has_resampled:
            return x, weights"""
            return NMDS_particles.resample_from_indexes(x, weights, indexes, optimizer)

        mask = np.zeros(x.shape[0], dtype=bool)
        mask[unique_indexes] = True

        x = x[mask]
        weights = np.ones(x.shape[0]) / x.shape[0]

        optimizer.update_states(mask)

        self.has_resampled = True

        return x, weights

    @staticmethod
    def update_weights(weights, k, f_evals):
        max_val = np.max(f_evals)
        weights = np.exp(-100 * f_evals / max_val) / weights + 1e-50
        weights[weights > 1e50] = 1e50
        weights /= np.sum(weights)
        return weights

    def optimize(self, function, sigma=-1, verbose=False):
        kernel = lambda x: rbf(x, sigma=sigma)

        dim = self.domain.shape[0]

        x = np.random.uniform(
            self.domain[:, 0], self.domain[:, 1], size=(self.n_particles, dim)
        )

        # random_indices = np.random.choice(x.shape[0], 5)
        # self.paths = [x[random_indices]]

        n_particles = self.n_particles

        self.has_resampled = False

        all_points = [x.copy()]
        for k in self.k_iter:
            optimizer = Adam(lr=self.lr) if self.adam else Static_Optimizer(lr=self.lr)
            weights = np.ones(n_particles) / n_particles

            for i in range(self.svgd_iter):
                logprob_grad_array = [np.zeros(dim)] * n_particles
                f_evals = [0] * n_particles

                for j in range(n_particles):
                    grad, f_eval = gradient(function, x[j])
                    logprob_grad_array[j] = -k * grad
                    f_evals[j] = f_eval

                f_evals = np.array(f_evals)

                # Move particles
                x = self.move(x, logprob_grad_array, kernel, optimizer)

                # Update weights
                weights = self.update_weights(weights, k, f_evals)

                # Resample if too few effective particles
                neff = self.neff(weights)
                """ print(weights)
                print(f_evals)
                print(n_particles)
                print(neff) """
                if neff < n_particles / 10:
                    # print("Resampling")
                    indexes = residual_resample(weights)
                    x, weights = self.resample_from_indexes(
                        x, weights, indexes, optimizer
                    )

                    n_particles = x.shape[0]

                # self.paths.append(x[random_indices])

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
