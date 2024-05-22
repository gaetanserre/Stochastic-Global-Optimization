#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from .__function__ import Function


class EggHolder(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.min = -959.6407

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0] / 2 + 47))) - x[
            0
        ] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
