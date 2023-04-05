#
# Created in 2023 by Gaëtan Serré
#
import numpy as np
from .__optimizer__ import Optimizer


class CMA_ES(Optimizer):
    def __init__(self, m_0, num_generations=10, lambda_=300, mu=10, lr=0.01, sigma=100):
        self.num_generations = num_generations
        self.lambda_ = lambda_
        self.mu = mu
        self.lr = lr
        self.sigma = sigma
        self.m_0 = m_0
        self.dim = m_0.shape[0]

    def update_mean(self, mean, x, weights):
        return mean + self.lr * np.sum(weights * (x[: self.mu] - mean), axis=0)

    def update_covariance(self, x, mean, weights):
        cov = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            cov += weights[i] * (x[i] - mean) @ (x[i] - mean).T
        return cov / self.sigma**2

    def optimize(self, function, verbose=False):
        mean = np.zeros(self.dim)
        cov = np.eye(self.dim)
        weights = np.array([self.mu - i + 1 for i in range(self.mu)])
        weights = (weights / np.sum(weights)).reshape(-1, 1)

        points = np.zeros((self.num_generations * self.lambda_, self.dim))
        values = np.zeros((self.num_generations * self.lambda_))

        for i in range(self.num_generations):
            x = np.random.multivariate_normal(mean, self.sigma**2 * cov, self.lambda_)
            y = np.array([function(xi) for xi in x])
            x = x[np.argsort(-y)]

            points[i * self.lambda_ : (i + 1) * self.lambda_] = x
            values[i * self.lambda_ : (i + 1) * self.lambda_] = y

            mean = self.update_mean(mean, x, weights)
            cov = self.update_covariance(x, mean, weights)

        best_idx = np.argmax(values)
        return (points[best_idx], values[best_idx]), points, values
