#
# Created in 2023 by Gaëtan Serré
#

import jax
from jax import jit
import jax.numpy as jnp
from scipy.spatial.distance import pdist, squareform
from .__optimizer__ import Optimizer
import time


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

    @staticmethod
    def static_step(
        grad, params, mask, betas, t, state_m, state_v, state_v_max, lr, eps
    ):
        grad = -grad

        state_m = betas[0] * state_m + (1 - betas[0]) * grad
        state_v = betas[1] * state_v + (1 - betas[1]) * grad**2

        m_hat = state_m / (1 - betas[0] ** t)
        v_hat = state_v / (1 - betas[1] ** t)

        """ if amsgrad:
            state_v_max = jnp.maximum(state_v_max, v_hat)
            return (
                state_m,
                state_v,
                state_v_max,
                lr,
                lr * m_hat / (jnp.sqrt(state_v_max) + eps),
            )
        else: """
        return (
            state_m,
            state_v,
            state_v_max,
            lr,
            params - mask.reshape(-1, 1) * (lr * m_hat / (jnp.sqrt(v_hat) + eps)),
        )

    def step(self, grad, params, mask):
        jit = jax.jit(self.static_step)
        self.t += 1

        self.state_m, self.state_v, self.state_v_max, self.lr, updated = jit(
            grad,
            params,
            mask,
            self.betas,
            self.t,
            self.state_m,
            self.state_v,
            self.state_v_max,
            self.lr,
            self.eps,
        )

        return updated


def rbf(x, h=-1):
    sq_dist = pdist(x)
    pairwise_dists = squareform(sq_dist) ** 2
    if h < 0:  # if h < 0, using median trick
        h = jnp.median(pairwise_dists) + 1e-10
        h = jnp.sqrt(0.5 * h / jnp.log(x.shape[0] + 1))

    # compute the rbf kernel
    Kxy = jnp.exp(-pairwise_dists / h**2 / 2)

    dxkxy = (x * Kxy.sum(axis=1).reshape(-1, 1) - Kxy @ x).reshape(
        x.shape[0], x.shape[1]
    ) / (h**2)

    return Kxy, dxkxy


def svgd(x, logprob_grad, kernel):
    Kxy, dxkxy = kernel(x)

    svgd_grad = (Kxy @ logprob_grad + dxkxy) / x.shape[0]
    return svgd_grad


class NMDS(Optimizer):
    def __init__(
        self,
        domain,
        n_particles,
        k_iter,
        svgd_iter,
        distance_q=0.1,
        value_q=0.5,
        lr=0.2,
    ):
        self.domain = domain
        self.n_particles = n_particles
        self.k_iter = k_iter
        self.svgd_iter = svgd_iter
        self.distance_q = distance_q
        self.value_q = value_q
        self.lr = lr

    def optimize(self, function, verbose=False):
        grad_f = jax.grad(function)

        kernel = rbf

        dim = self.domain.shape[0]

        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(
            key,
            shape=(self.n_particles, dim),
            minval=self.domain[:, 0],
            maxval=self.domain[:, 1],
        )

        @jit
        def get_grad(x, mask):
            logprob_grad_array = [jnp.zeros(dim)] * x.shape[0]
            f_evals = [1e10] * x.shape[0]
            for j in range(x.shape[0]):
                logprob_grad_array[j] = -k * grad_f(x[j]) * mask[j]
                f_evals[j] = function(x[j]) + (1 - mask[j]) * 1e10
            logprob_grad_array = jnp.array(logprob_grad_array)
            f_evals = jnp.array(f_evals)
            return logprob_grad_array, f_evals

        @jit
        def remove_particles(x, x_new, x_values):
            distance = jnp.linalg.norm(x - x_new, axis=1)
            dist_quantile = jnp.quantile(distance, q=self.distance_q)
            dist_mask = distance < dist_quantile

            value_quantile = jnp.quantile(x_values, q=self.value_q)
            value_mask = x_values > value_quantile

            mask = ~(dist_mask & value_mask)
            return mask

        mask = jnp.ones(x.shape[0], dtype=bool)
        all_points = [x.copy()]
        for k in self.k_iter:
            optimizer = Adam(lr=self.lr)
            for i in range(self.svgd_iter):
                logprob_grad_array, f_evals = get_grad(x, mask)

                svgd_grad = svgd(x, logprob_grad_array, kernel)
                x_new = optimizer.step(svgd_grad, x, mask)

                # clamp to domain
                x_new = jnp.clip(x_new, self.domain[:, 0], self.domain[:, 1])

                if x_new.shape[0] > 10:
                    mask = remove_particles(x, x_new, f_evals)

                x = x_new

                # save all points
                all_points.append(x.copy())

        evals = jnp.array([function(xi) for xi in x]).flatten()
        best_idx = jnp.argmin(evals)
        min_eval = evals[best_idx]
        best_particle = x[best_idx]
        if verbose:
            print(f"Best particle found: {best_particle}. Eval at f(best): {min_eval}.")

        np_all_points = None
        for i, np_seq in enumerate(all_points):
            if i == 0:
                np_all_points = np_seq
            else:
                np_all_points = jnp.concatenate((np_all_points, np_seq), axis=0)

        all_evals = jnp.array([function(xi) for xi in np_all_points]).flatten()
        return (best_particle, min_eval), np_all_points, all_evals

    def optimize_jit(self, function, verbose):
        optimize = jax.jit(self.optimize)
        return optimize(function, verbose)
