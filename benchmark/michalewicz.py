#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Michalewicz(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.min = -9.66015

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        dim = x.shape[0]
        id_ = np.arange(1, dim + 1)
        return -np.sum(np.sin(x) * np.sin(id_ * x**2 / np.pi) ** (2 * 10))
