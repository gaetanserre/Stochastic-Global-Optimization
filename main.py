#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
import os

# benchmark functions
from benchmark.ackley import Ackley
from benchmark.himmelblau import Himmelblau
from benchmark.holder import Holder
from benchmark.rastrigin import Rastrigin
from benchmark.rosenbrock import Rosenbrock
from benchmark.sphere import Sphere
from benchmark.cos import Cos
from benchmark.branin import Branin
from benchmark.eggholder import EggHolder
from benchmark.drop_wave import Drop_Wave
from benchmark.michalewicz import Michalewicz
from benchmark.goldstein_price import Goldstein_Price
from optims.extended_function import extended_function

from benchmark.epidemio.simulation import Simulation

# Optimizations algorithms
from optims.PRS import PRS
from optims.AdaLIPO_E import AdaLIPO_E
from optims.CMA_ES import CMA_ES
from optims.NMDS_particles import NMDS


def time_it(function, args={}):
    import time

    start = time.time()
    ret = function(**args)
    end = time.time()
    return ret, end - start


def print_color(str, color):
    print(f"\033[{color}m" + str + "\033[0m")


print_yellow = lambda str: print_color(str, 93)
print_blue = lambda str: print_color(str, 94)
print_green = lambda str: print_color(str, 92)


def match_optim(optim_cls, bounds, num_evals, is_sim=False):
    if optim_cls == PRS:
        return optim_cls(bounds, num_evals=num_evals)
    elif optim_cls == AdaLIPO_E:
        return optim_cls(bounds, max_evals=num_evals)
    elif optim_cls == CMA_ES:
        m_0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
        return optim_cls(
            bounds,
            m_0,
            num_generations=num_evals // 5,
            lambda_=5,
            cov_method="full",
        )
    elif optim_cls == NMDS:
        return optim_cls(
            bounds,
            n_particles=100,
            k_iter=[1000],
            svgd_iter=500,
            lr=0.1 if is_sim else 0.2,
        )
    else:
        raise NotImplementedError(f"{optim_cls} not implemented.")


if __name__ == "__main__":
    num_exp = 10

    functions = [Michalewicz()]

    bounds = [
        np.array(
            [
                (0, np.pi),
                (0, np.pi),
            ]
        ),
        np.array(
            [
                (-4, 4),
                (-4, 4),
            ]
        ),
        np.array(
            [
                (-10, 10),
                (-10, 10),
            ]
        ),
        np.array(
            [
                (-5.12, 5.12),
                (-5.12, 5.12),
            ]
        ),
        np.array(
            [
                (-3, 3),
                (-3, 3),
            ]
        ),
        np.array(
            [
                (-10, 10),
                (-10, 10),
            ]
        ),
    ]

    optimizers_cls = [NMDS]

    num_eval = 2000

    for i, function in enumerate(functions):
        print_yellow(f"Function: {function.__class__.__name__}.")

        dim = bounds[i].shape[0]
        plot_figures = dim <= 2
        if plot_figures:
            from fig_generator import FigGenerator

            fig_gen = FigGenerator(function, bounds[i])

            if not os.path.exists("figures"):
                os.makedirs("figures")

        for optimizer_cls in optimizers_cls:
            print_blue(f"Optimizer: {optimizer_cls.__name__}.")

            optimizer = match_optim(
                optimizer_cls,
                bounds[i],
                num_eval,
                is_sim=function.__class__.__name__ == "Simulation",
            )

            times = []
            best_values = []
            num_evals = 0

            for _ in range(num_exp):
                function.n = 0
                ret, time = time_it(
                    optimizer.optimize,
                    {
                        "function": function,
                        "verbose": False,
                    },
                )
                best_point, points, values = ret
                num_evals += function.n

                best_point = (best_point[0], function(best_point[0]))

                print(f"Time: {time:.4f}s. Best point found: {best_point}.")

                times.append(time)
                best_values.append(best_point[1])
            print_green(
                f"Average time: {np.mean(times):.4f} +- {np.std(times):.2f}s. Average number of evaluations: {num_evals / num_exp:.2f}. Average best value: {np.mean(best_values):.4f} +- {np.std(best_values):.2f}.\n"
            )
            if plot_figures:
                path = f"figures/{type(function).__name__}_{optimizer_cls.__name__}.svg"
                fig_gen.gen_figure(points, values, path=path)
