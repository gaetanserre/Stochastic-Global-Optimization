#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Drop_Wave(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        return -(1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))) / (
            0.5 * (x[0] ** 2 + x[1] ** 2) + 2
        )
