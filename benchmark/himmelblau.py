#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class Himmelblau(Function):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray) -> float:
        return -((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2)
