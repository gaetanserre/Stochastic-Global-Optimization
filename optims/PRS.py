#
# Created in 2023 by Gaëtan Serré
#

from .__optimizer__ import Optimizer
import numpy as np


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


class PRS(Optimizer):
    """
    This class implements the Pure Random Searh algorithm.
    """

    def __init__(self, bounds, num_evals):
        self.bounds = bounds
        self.num_evals = num_evals

    def optimize(self, function, verbose=False):
        values = np.zeros(self.num_evals)
        points = np.zeros((self.num_evals, self.bounds.shape[0]))

        for i in range(self.num_evals):
            points[i] = Uniform(self.bounds)
            values[i] = function(points[i])

        best_idx = np.argmax(values)
        return (points[best_idx], values[best_idx]), points, values
