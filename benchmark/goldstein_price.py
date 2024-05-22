#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Goldstein_Price(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.min = 3

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        return (
            1
            + (x[0] + x[1] + 1) ** 2
            * (
                19
                - 14 * x[0]
                + 3 * x[0] ** 2
                - 14 * x[1]
                + 6 * x[0] * x[1]
                + 3 * x[1] ** 2
            )
        ) * (
            30
            + (2 * x[0] - 3 * x[1]) ** 2
            * (
                18
                - 32 * x[0]
                + 12 * x[0] ** 2
                + 48 * x[1]
                - 36 * x[0] * x[1]
                + 27 * x[1] ** 2
            )
        )
