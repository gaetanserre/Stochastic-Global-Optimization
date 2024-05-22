#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Camel(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.min = -1.0316

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        return (
            (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2
            + x[0] * x[1]
            + (-4 + 4 * x[1] ** 2) * x[1] ** 2
        )
