#
# Created in 2024 by Gaëtan Serré
#

from .__function__ import Function
import numpy as np


class Michalewicz(Function):
    def __init__(self, d) -> None:
        super().__init__()
        self.n = 0
        if d == 2:
            self.min = -1.8013
        elif d == 5:
            self.min = -4.687658
        elif d == 10:
            self.min = -9.66015
        else:
            x = np.random.uniform(0, np.pi, size=(10_000_000, d))
            y = np.array([self(x_) for x_ in x])
            self.min = np.min(y)

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        dim = x.shape[0]
        id_ = np.arange(1, dim + 1)
        return -np.sum(np.sin(x) * np.sin(id_ * x**2 / np.pi) ** (2 * 10))
