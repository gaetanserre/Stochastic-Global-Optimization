#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Ackley(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.min = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        a = 20
        b = 0.2
        c = 1
        return (
            -a * np.exp(-b * np.sqrt(np.sum(x**2) / len(x)))
            - np.exp(np.sum(np.cos(c * x)) / len(x))
            + a
            + np.e
        )
