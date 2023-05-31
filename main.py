#
# Created in 2023 by Gaëtan Serré
#

import numpy as np

# benchmark functions
from benchmark.rastrigin import Rastrigin
from benchmark.square import Square
from benchmark.rosenbrock import Rosenbrock
from benchmark.holder import Holder
from benchmark.cos import Cos

from benchmark.epidemio.simulation import Simulation

# Optimizations algorithms
from optims.PRS import PRS
from optims.AdaLIPO_E import AdaLIPO_E
from optims.CMA_ES import CMA_ES
from optims.GO_SVGD import GO_SVGD


def time_it(function, args={}):
    import time

    start = time.time()
    ret = function(**args)
    end = time.time()
    return ret, end - start


def print_yellow(str):
    print("\033[93m" + str + "\033[0m")


def print_blue(str):
    print("\033[94m" + str + "\033[0m")


def print_green(str):
    print("\033[92m" + str + "\033[0m")


if __name__ == "__main__":
    num_exp = 5

    functions = [Simulation()]

    bounds = [np.array([(0, 1), (0, 1), (0, 1)])]

    optimizers_cls = [PRS, AdaLIPO_E, CMA_ES, GO_SVGD]

    num_eval = 100

    for i, function in enumerate(functions):
        print_yellow(f"Function: {function.__class__.__name__}.")
        for optimizer_cls in optimizers_cls:
            print_blue(f"Optimizer: {optimizer_cls.__name__}.")

            if optimizer_cls == PRS:
                optimizer = optimizer_cls(bounds[i], num_evals=num_eval)
            elif optimizer_cls == AdaLIPO_E:
                optimizer = optimizer_cls(bounds[i], max_evals=num_eval)
            elif optimizer_cls == CMA_ES:
                m_0 = np.random.uniform(bounds[i][:, 0], bounds[i][:, 1])
                optimizer = optimizer_cls(
                    bounds[i],
                    m_0,
                    num_generations=num_eval // 5,
                    lambda_=5,
                    cov_method="full",
                )
            elif optimizer_cls == GO_SVGD:
                optimizer = optimizer_cls(
                    bounds[i],
                    n_particles=10,
                    k_iter=[50, 250],
                    svgd_iter=10,
                )
            else:
                raise NotImplementedError(f"{optimizer_cls} not implemented.")

            times = []
            best_values = []
            evals = []

            for _ in range(num_exp):
                ret, time = time_it(
                    optimizer.optimize, {"function": function, "verbose": False}
                )
                best_point, points, values = ret

                print(
                    f"Time: {time:.4f}s. Best point found: {best_point}. Num evals {len(values)}."
                )

                times.append(time)
                best_values.append(best_point[1])
                evals.append(len(values))
            print_green(
                f"Average time: {np.mean(times):.4f} +- {np.std(times):.2f}s. Average best value: {np.mean(best_values):.4f} +- {np.std(best_values):.2f}. Average num eval: {np.mean(evals):.4f} +- {np.std(evals):.2f}.\n"
            )
