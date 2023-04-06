#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Rosenbrock(Function):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray) -> float:
        return -np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
