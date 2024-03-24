#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Holder(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.min = -19.2085

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        return -np.abs(
            np.sin(x[0])
            * np.cos(x[1])
            * np.exp(np.abs(1 - (np.sqrt(x[0] ** 2 + x[1] ** 2)) / np.pi))
        )
