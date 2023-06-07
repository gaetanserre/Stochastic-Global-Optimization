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
        eps = 1e-10
        mu = 0.01
        penalties_inf = np.sum(
            [np.log(-(bounds[i, 0] - x[i]) + eps) for i in range(bounds.shape[0])]
        )
        penalties_sup = np.sum(
            [np.log(-(x[i] - bounds[i, 1]) + eps) for i in range(bounds.shape[0])]
        )

        return f(x) + mu * (penalties_inf + penalties_sup)

    return extended_f
