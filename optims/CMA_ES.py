#
# Created in 2023 by Gaëtan Serré
#
import numpy as np
from .__optimizer__ import Optimizer


class CMA_ES(Optimizer):
    """
    This class implements the Covariance Matrix Adaptation Evolution Strategy algorithm.
    """

    def __init__(
        self,
        bounds,
        m_0,
        num_generations=100,
        lambda_=10,
        mu=5,
        lr=0.01,
        sigma=100,
        cov_method="scratch",
    ):
        self.m_0 = m_0
        self.dim = m_0.shape[0]
        self.bounds = bounds
        self.num_generations = num_generations
        self.lambda_ = lambda_
        self.mu = mu
        self.lr = lr
        self.sigma = sigma
        self.cov_method = cov_method

    def update_mean(self, mean, x, weights):
        return mean + self.lr * np.sum(weights * (x[: self.mu] - mean), axis=0)

    def update_covariance_scratch(self, x, mean, weights):
        cov = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            cov += weights[i] * (x[i] - mean) @ (x[i] - mean).T
        return cov / self.sigma**2

    def update_covariance_rank_mu(self, x, mean, weights, cov):
        mu_eff = 1 / np.sum(weights**2)
        c_mu = min(1, mu_eff / self.dim**2)

        new_cov = self.update_covariance_scratch(x, mean, weights)
        new_cov *= c_mu
        new_cov += (1 - c_mu) * cov
        return new_cov

    def optimize(self, function, verbose=False):
        mean = np.zeros(self.dim)
        cov = np.eye(self.dim)
        weights = np.array([self.mu - i + 1 for i in range(self.mu)])
        weights = (weights / np.sum(weights)).reshape(-1, 1)

        points = np.zeros((self.num_generations * self.lambda_, self.dim))
        values = np.zeros((self.num_generations * self.lambda_))

        for i in range(self.num_generations):
            x = np.random.multivariate_normal(mean, self.sigma**2 * cov, self.lambda_)
            # clip to bounds
            x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
            y = np.array([function(xi) for xi in x])
            x_sorted = x[np.argsort(-y)]

            points[i * self.lambda_ : (i + 1) * self.lambda_] = x
            values[i * self.lambda_ : (i + 1) * self.lambda_] = y

            if self.cov_method == "rank_mu":
                cov = self.update_covariance_rank_mu(x_sorted, mean, weights, cov)
            elif self.cov_method == "scratch":
                cov = self.update_covariance_scratch(x_sorted, mean, weights)
            else:
                raise ValueError(f"Unknown covariance method {self.cov_method}")
            mean = self.update_mean(mean, x_sorted, weights)

        best_idx = np.argmax(values)
        return (points[best_idx], values[best_idx]), points, values
