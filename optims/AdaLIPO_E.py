#
# Created in 2023 by Gaëtan Serré
#

from .__optimizer__ import Optimizer
import numpy as np
from collections import deque


def Uniform(X: np.array):
    """
    This function generates a random point in the feasible region X. We assume that X is a subset of R^n
    described by the inequalities X = {x in R^n | a_i <= x_i <= b_i, i = 0, ..., m-1} where a_i, b_i are given
    such that X[i,j] = [a_i, b_i] for i = 0, ..., m-1 and j = 0, 1.
    For simplicity, we assume that X C Rectangle given by an infinite norm (i.e. X = {x in R^n | -M <= x_i <= M, i = 1, ..., n}).
    X: feasible region (numpy array)
    """

    theta = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        theta[i] = np.random.uniform(X[i, 0], X[i, 1])
    return theta


def Bernoulli(p: float):
    """
    This function generates a random variable following a Bernoulli distribution.
    p: probability of success (float)
    """
    a = np.random.uniform(0, 1)
    if a <= p:
        return 1
    else:
        return 0


def slope_stop_condition(last_nb_samples, size_slope, max_slope):
    """
    Check if the slope of the last `size_slope` points of the the nb_samples vs nb_evaluations curve
    is greater than max_slope.
    """
    if len(last_nb_samples) == size_slope:
        slope = (last_nb_samples[-1] - last_nb_samples[0]) / (len(last_nb_samples) - 1)
        return slope > max_slope
    else:
        return False


class AdaLIPO_E(Optimizer):
    def __init__(self, bounds, max_iter, window_slope=5, max_slope=700):
        self.bounds = bounds
        self.max_iter = max_iter
        self.window_slope = window_slope
        self.max_slope = max_slope

    def return_process(self, points, values, nb_samples):
        points = points[:nb_samples]
        values = values[:nb_samples]
        best_idx = np.argmax(values)
        return (points[best_idx], values[best_idx]), points, values

    def optimize(self, function, verbose=False):
        t = 1
        alpha = 10e-2
        k_hat = 0

        X_1 = Uniform(self.bounds)
        nb_samples = 1

        # We keep track of the last `size_slope` values of nb_samples to compute the slope
        last_nb_samples = deque([1], maxlen=self.window_slope)

        points = np.zeros((self.max_iter, X_1.shape[0]))
        values = np.zeros(self.max_iter)
        points[0] = X_1
        values[0] = function(X_1)

        def k(i):
            """
            Series of potential Lipschitz constants.
            """
            return (1 + alpha) ** i

        def p(t):
            """
            Probability of success for exploration/exploitation.
            """
            if t == 1:
                return 1
            else:
                return 1 / np.log(t)

        def condition(x, values, k, points, iter):
            """
            Subfunction to check the condition in the loop, depending on the set of values we already have.
            values: set of values of the function we explored (numpy array)
            x: point to check (numpy array)
            k: Lipschitz constant (float)
            points: set of points we have explored (numpy array)
            """
            max_val = np.max(values)

            left_min = np.min(
                values[:iter] + k * np.linalg.norm(x - points[:iter], ord=2, axis=1)
            )

            return left_min >= max_val

        # Main loop
        ratios = []
        while t < self.max_iter:
            B_tp1 = Bernoulli(p(t))
            if B_tp1 == 1:
                # Exploration
                X_tp1 = Uniform(self.bounds)
                nb_samples += 1
                last_nb_samples[-1] = nb_samples
                points[t] = X_tp1
                value = function(X_tp1)
            else:
                # Exploitation
                while True:
                    X_tp1 = Uniform(self.bounds)
                    nb_samples += 1
                    last_nb_samples[-1] = nb_samples
                    if condition(X_tp1, values, k_hat, points, t):
                        points[t] = X_tp1
                        break
                    elif slope_stop_condition(
                        last_nb_samples, self.window_slope, self.max_slope
                    ):
                        print(
                            f"Exponential growth of the number of samples. Stopping the algorithm at iteration {t}."
                        )
                        return self.return_process(points, values, t)
                value = function(X_tp1)

            values[t] = value
            for i in range(t):
                ratios.append(
                    np.abs(value - values[i]) / np.linalg.norm(X_tp1 - points[i], ord=2)
                )

            i_hat = int(np.ceil(np.log(max(ratios)) / np.log(1 + alpha)))
            k_hat = k(i_hat)

            t += 1
            last_nb_samples.append(0)

            if t % 200 == 0 and verbose:
                print(
                    f"Iteration: {t} Lipschitz constant: {k_hat:.4f} Number of samples: {nb_samples}"
                )

        return self.return_process(points, values, t)
