#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Rosenbrock(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.min = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
