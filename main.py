#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from benchmark.rastrigin import Rastrigin
from benchmark.square import Square
from optims.AdaLIPO_E import AdaLIPO_E
from optims.CMA_ES import CMA_ES

if __name__ == "__main__":
    functions = [Rastrigin(), Square()]
    bounds = [
        np.array([(-5.12, 5.12), (-5.12, 5.12)]),
        np.array([(-10, 10), (-10, 10)]),
    ]

    optimizers_cls = [AdaLIPO_E, CMA_ES]

    for i, function in enumerate(functions):
        print(f"Function: {function.__class__.__name__}.")
        for optimizer_cls in optimizers_cls:
            if optimizer_cls == AdaLIPO_E:
                optimizer = optimizer_cls(bounds[i], max_iter=3000)
            else:
                optimizer = optimizer_cls(np.zeros(bounds[i].shape[0]))

            print(f"Optimizer: {optimizer_cls.__name__}.")
            best_point, points, values = optimizer.optimize(function, verbose=True)
            print(f"Best point found: {best_point}. Num evals {len(values)}\n")
