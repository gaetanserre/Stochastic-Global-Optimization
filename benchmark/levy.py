#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Levy(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.min = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        w = 1 + (x - 1) / 4
        return (
            np.sin(np.pi * w[0]) ** 2
            + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
            + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        )
