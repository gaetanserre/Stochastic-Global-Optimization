#
# Created in 2023 by Gaëtan Serré
#

import numpy as np


class Function:
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError
