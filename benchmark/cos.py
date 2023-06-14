#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Cos(Function):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray) -> float:
        return -(np.cos(x[0] ** 2) + x[0] / 5 + 1)
