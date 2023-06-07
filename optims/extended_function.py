#
# Created in 2023 by Gaëtan Serré
#

import numpy as np


def extended_function(f, bounds):
    """
    This function is used to extend the function to the whole space.
    It decreases exponentially to infinity outside the bounds.
    """

    def extended_f(x):
        penalties_inf = np.sum(
            [
                np.log(1 / np.maximum(1, bounds[i, 0] - x[i]) ** 20)
                for i in range(bounds.shape[0])
            ]
        )
        penalties_sup = np.sum(
            [
                np.log(1 / np.maximum(1, x[i] - bounds[i, 1]) ** 20)
                for i in range(bounds.shape[0])
            ]
        )
        return f(x) + penalties_inf + penalties_sup

    return extended_f
