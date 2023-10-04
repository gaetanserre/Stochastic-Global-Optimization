#
# Created in 2023 by Gaëtan Serré
#
import numpy as np
import scipy
from .__optimizer__ import Optimizer


class CMA_ES(Optimizer):
    """
    This class implements the Covariance Matrix Adaptation Evolution Strategy algorithm.
    """

    def __init__(
        self,
        domain,
        m_0,
        num_generations=100,
        lambda_=None,
        mu=None,
        cov_method="full",
    ):
        self.m_0 = m_0
        self.dim = m_0.shape[0]
        self.domain = domain
        self.num_generations = num_generations
        self.lambda_ = (
            lambda_ if lambda_ is not None else 4 + int(np.floor(3 * np.log(self.dim)))
        )
        self.mu = self.lambda_ // 2 if mu is None else mu
        self.cov_method = cov_method

    def update_mean(self, mean, x, weights):
        c_m = 1
        return mean + c_m * np.sum(weights[: self.mu] * (x[: self.mu] - mean), axis=0)

    def sum_covariance(self, x, mean, weights, endpoint):
        cov = np.zeros((self.dim, self.dim))
        for i in range(endpoint):
            cov += weights[i] * (x[i] - mean).T @ (x[i] - mean)
        return cov

    def update_covariance_scratch(self, x, mean, weights, sigma):
        cov = self.sum_covariance(x, mean, weights, self.mu)
        return cov / sigma**2

    def update_covariance_rank_mu(self, x, mean, weights, cov, sigma):
        mu_eff = 1 / np.sum(weights[: self.mu] ** 2)
        c_mu = min(1, mu_eff / self.dim**2)

        new_cov = self.update_covariance_scratch(x, mean, weights, sigma)
        new_cov *= c_mu
        new_cov += (1 - c_mu) * cov
        return new_cov

    def update_covariance_rank_1_mu(self, x, mean, weights, cov, p_c, sigma):
        mu_eff = 1 / np.sum(weights[: self.mu] ** 2)
        alpha_cov = 2
        c_1 = alpha_cov / ((self.dim + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1 - c_1,
            alpha_cov
            * (0.25 + mu_eff + (1 / mu_eff) - 2)
            / ((self.dim + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)

        coeff_1 = (1 - c_1 - c_mu * np.sum(weights)) * cov

        rank_mu = (
            c_mu * self.sum_covariance(x, mean, weights, self.lambda_) / sigma**2
        )

        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * np.sum(
            weights[: self.mu] * (x[: self.mu] - mean), axis=0
        ) / sigma

        rank_1 = c_1 * p_c.reshape(-1, 1) @ p_c.reshape(1, -1)

        return coeff_1 + rank_mu + rank_1, p_c

    def update_step_size(self, x, mean, sigma, p_sigma, cov, weights):
        mu_eff = 1 / np.sum(weights[: self.mu] ** 2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma

        try:
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * np.linalg.inv(scipy.linalg.sqrtm(cov)) @ np.sum(
                weights[: self.mu] * (x[: self.mu] - mean), axis=0
            ) / sigma
        except Exception as e:
            print(np.sum(weights[: self.mu] * (x[: self.mu] - mean), axis=0) / sigma)
            print(cov)
            raise e

        sigma = sigma * np.exp(
            (c_sigma / d_sigma) * ((np.linalg.norm(p_sigma) / np.sqrt(self.dim)) - 1)
        )

        sigma = min(sigma, 1e5)

        return sigma, p_sigma

    def create_weights(self):
        # Intermediate weights
        weights = np.zeros(self.lambda_)
        for i in range(self.lambda_):
            weights[i] = np.log((self.lambda_ + 1) / 2) - np.log(i + 1)

        mu_eff_bar = np.sum(weights) ** 2 / np.sum(weights**2)

        # Normalization of the mu weights
        weights[: self.mu] /= np.sum(weights[: self.mu])

        mu_eff = 1 / np.sum(weights[: self.mu] ** 2)
        alpha_cov = 2
        c_1 = alpha_cov / ((self.dim + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1 - c_1,
            alpha_cov
            * (0.25 + mu_eff + (1 / mu_eff) - 2)
            / ((self.dim + 2) ** 2 + alpha_cov * mu_eff / 2),
        )

        alpha_mu = 1 + c_1 / c_mu
        alpha_mu_eff_bar = 1 + (2 * mu_eff_bar) / (mu_eff + 2)
        alpha_pos_def = ((1 - c_1 - c_mu) / self.dim) * c_mu

        weights[self.mu :] = min(alpha_mu, alpha_mu_eff_bar, alpha_pos_def) / np.sum(
            weights[self.mu :]
        )

        return weights.reshape(-1, 1)

    def optimize(self, function, verbose=False):
        mean = self.m_0
        cov = np.eye(self.dim)
        p_c = np.zeros(self.dim)
        p_sigma = np.zeros(self.dim)
        sigma = 1

        weights = self.create_weights()

        points = np.zeros((self.num_generations * self.lambda_, self.dim))
        values = np.zeros((self.num_generations * self.lambda_))

        for i in range(self.num_generations):
            x = mean + sigma * np.random.multivariate_normal(
                np.zeros(self.dim), cov, self.lambda_
            )

            # clip the points to the domain
            x = np.clip(x, self.domain[:, 0], self.domain[:, 1])

            y = np.array([function(xi) for xi in x])
            x_sorted = x[np.argsort(y)]

            points[i * self.lambda_ : (i + 1) * self.lambda_] = x
            values[i * self.lambda_ : (i + 1) * self.lambda_] = y

            eigvals = np.linalg.eigvals(cov)
            cond = np.max(eigvals) / np.min(eigvals) if np.min(eigvals) > 0 else np.inf
            if cond > 1e14:
                break

            if self.cov_method == "rank_mu":
                cov = self.update_covariance_rank_mu(
                    x_sorted, mean, weights, cov, sigma
                )
            elif self.cov_method == "scratch":
                cov = self.update_covariance_scratch(x_sorted, mean, weights, sigma)
            elif self.cov_method == "full":
                cov, p_c = self.update_covariance_rank_1_mu(
                    x_sorted, mean, weights, cov, p_c, sigma
                )
            else:
                raise ValueError(f"Unknown covariance method {self.cov_method}")

            if self.cov_method == "full":
                try:
                    sigma, p_sigma = self.update_step_size(
                        x_sorted, mean, sigma, p_sigma, cov, weights
                    )
                except:
                    break

            mean = self.update_mean(mean, x_sorted, weights)

        best_idx = np.argmin(values[: i * self.lambda_])
        return (
            (points[best_idx], values[best_idx]),
            points[: i * self.lambda_],
            values[: i * self.lambda_],
        )
