#
# Created in 2023 by Gaëtan Serré
#

from .__optimizer__ import Optimizer
import numpy as np


class PRS(Optimizer):
    """
    This class implements the Pure Random Searh algorithm.
    """

    def __init__(self, domain, num_evals):
        self.domain = domain
        self.num_evals = num_evals

    def optimize(self, function, verbose=False):
        points = np.random.uniform(
            self.domain[:, 0],
            self.domain[:, 1],
            size=(self.num_evals, self.domain.shape[0]),
        )

        values = np.array([function(point) for point in points])

        best_idx = np.argmin(values)
        return (points[best_idx], values[best_idx]), points, values
