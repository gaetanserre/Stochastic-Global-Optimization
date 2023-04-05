#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Rastrigin(Function):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray) -> float:
        dim = x.shape[0]
        return -(10 * dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))
