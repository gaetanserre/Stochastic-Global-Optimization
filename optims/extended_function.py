#
# Created in 2023 by Gaëtan Serré
#

import numpy as np


def extended_function(f, bounds):
    """
    This function is used to extend the function to the whole space.
    It increases exponentially to infinity outside the bounds.
    """

    def extended_f(x):
        eps = 1e-2
        mu = 1e-6
        penalties_inf = np.sum(
            [np.log(-(bounds[i, 0] - x[i]) + eps) for i in range(bounds.shape[0])]
        )
        penalties_sup = np.sum(
            [np.log(-(x[i] - bounds[i, 1]) + eps) for i in range(bounds.shape[0])]
        )

        # x_old = x.copy()
        # x = np.array([0, 0])

        """ print("Begin bounds")
        print(bounds[0, 0] - x[0])
        print(bounds[1, 0] - x[0])
        print(bounds[0, 1] - x[0])
        print(bounds[1, 1] - x[0])
        print("End bounds") """

        if np.linalg.norm(bounds - x) < 0.1:
            penalties_inf = np.sum(
                [
                    np.exp(1 / (-(bounds[i, 0] - x[i]) + eps))
                    for i in range(bounds.shape[0])
                ]
            )
            penalties_sup = np.sum(
                [
                    np.exp(1 / (-(x[i] - bounds[i, 1]) + eps))
                    for i in range(bounds.shape[0])
                ]
            )

            # print((penalties_inf + penalties_sup))
            return f(x) + mu * (penalties_inf + penalties_sup)
        else:
            return f(x)

    return extended_f
