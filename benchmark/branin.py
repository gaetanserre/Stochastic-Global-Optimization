#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Branin(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.min = 0.397887

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        a = 1
        b = 5.1 / (4 * np.pi) ** 2
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return (
            a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2
            + s * (1 - t) * np.cos(x[0])
            + s
        )
