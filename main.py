#
# Created in 2023 by Gaëtan Serré
#

from benchmark.rastrigin import Rastrigin
from benchmark.square import Square
from optims.AdaLIPO_E import AdaLIPO_E
import numpy as np

if __name__ == "__main__":
    function = Square()
    bounds = np.array([(-10, 10), (-10, 10), (-10, 10), (-10, 10)])
    optimizer = AdaLIPO_E(bounds, max_iter=10_000)
    best_point, points, values = optimizer.optimize(function, verbose=True)
    print(f"Best point found: {best_point}.")
