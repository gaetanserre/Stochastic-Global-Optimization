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
from optims.NMDS import NMDS
from optims.NMDS_particles import NMDS_particles
from optims.NMDS_particles_SIR import NMDS_particles_SIR
from optims.NMDS_CMA import NMDS_CMA
from optims.BayesOpt import BayesOpt
from optims.N_CMA_ES import CMA_ES
from optims.WOA import WOA
from optims.Simulated_Annealing import SimulatedAnnealing

from pretty_printer import pprint_results_get_rank, pprint_rank


def time_it(function, args={}):
    import time

    start = time.time()
    ret = function(**args)
    end = time.time()
    return ret, end - start


def print_color(str, color):
    print(f"\033[{color}m" + str + "\033[0m")


print_pink = lambda str: print_color(str, 95)
print_blue = lambda str: print_color(str, 94)
print_green = lambda str: print_color(str, 92)


def match_optim(optim_cls, bounds, num_evals, is_sim=False):
    if optim_cls == PRS:
        return optim_cls(bounds, num_evals=num_evals)
    elif optim_cls == AdaLIPO_E:
        return optim_cls(bounds, max_evals=num_evals)
    elif optim_cls == NMDS or optim_cls == NMDS_particles_SIR:
        return optim_cls(
            bounds,
            n_particles=500,
            k_iter=[10_000],
            svgd_iter=100,
            lr=0.1 if is_sim else 0.2,
        )
    elif optim_cls == NMDS_particles:
        return optim_cls(
            bounds,
            n_particles=5000,
            k_iter=[10_000],
            svgd_iter=300,
            cma_iter=30_000,
            sigma=-1,
            lr=0.1 if is_sim else 0.2,
        )
    elif optim_cls == NMDS_CMA:
        return optim_cls(
            bounds,
            n_particles=500,
            k_iter=[1000],
            svgd_iter=1000,
            cma_iter=200,
            lr=0.1 if is_sim else 0.2,
        )
    elif optim_cls == BayesOpt:
        return optim_cls(bounds, n_iter=70)
    elif optim_cls == CMA_ES:
        m_0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
        return optim_cls(bounds, m_0, max_evals=num_evals)
    elif optim_cls == WOA:
        return optim_cls(bounds, n_gen=30, n_sol=num_evals // 30)
    elif optim_cls == SimulatedAnnealing:
        return optim_cls(bounds, num_evals)
    else:
        raise NotImplementedError(f"{optim_cls} not implemented.")


def merge_ranks(rank_dict, new_ranks):
    if rank_dict == {}:
        return new_ranks
    else:
        for k, v in rank_dict.items():
            rank_dict[k] = v + new_ranks[k]
        return rank_dict


def create_bounds(min, max, dim):
    bounds = [(min, max) for _ in range(dim)]
    return np.array(bounds)


if __name__ == "__main__":
    num_exp = 10

    functions = {
        "Ackley": [Ackley(), create_bounds(-32.768, 32.768, 50)],
        # "Branin": [Branin(), np.array([(-5, 10), (0, 15)])],
        # "Drop Wave": [Drop_Wave(), create_bounds(-5.12, 5.12, 2)],
        # "Egg Holder": [EggHolder(), create_bounds(-512, 512, 2)],
        # "Goldstein Price": [Goldstein_Price(), create_bounds(-2, 2, 2)],
        # "Himmelblau": [Himmelblau(), create_bounds(-4, 4, 2)],
        # "Holder Table": [Holder(), create_bounds(-10, 10, 2)],
        # "Michalewicz": [Michalewicz(), create_bounds(0, np.pi, 50)],
        "Rastrigin": [Rastrigin(), create_bounds(-5.12, 5.12, 50)],
        # "Rosenbrock": [Rosenbrock(), create_bounds(-3, 3, 50)],
        "Sphere": [Sphere(), create_bounds(-10, 10, 50)],
    }

    """ optimizers_cls = [
        AdaLIPO_E,
        BayesOpt,
        WOA,
        CMA_ES,
        NMDS,
        NMDS_particles,
        NMDS_particles_old,
    ] """
    # optimizers_cls = [WOA, CMA_ES, NMDS, NMDS_particles]
    # optimizers_cls = [WOA, CMA_ES, NMDS, NMDS_particles]
    optimizers_cls = [NMDS_particles]

    # num_evals = [2000, 70, 200_000, 200_000, 0, 0, 0]
    # num_evals = [600_000, 600_000, 0, 0, 0]
    num_evals = [200_000, 0]

    all_ranks = {}

    for function_name, objects in functions.items():
        print_pink(f"Function: {function_name}.")

        function = objects[0]
        bounds = objects[1]

        dim = bounds.shape[0]
        plot_figures = False  # dim <= 2
        if plot_figures:
            from fig_generator import FigGenerator

            fig_gen = FigGenerator(function, bounds)

            if not os.path.exists("figures"):
                os.makedirs("figures")

        opt_dict = {}
        for i, optimizer_cls in enumerate(optimizers_cls):
            print_blue(f"Optimizer: {optimizer_cls.__name__}.")

            optimizer = match_optim(
                optimizer_cls,
                bounds,
                num_evals[i],
                is_sim=function.__class__.__name__ == "Simulation",
            )

            times = []
            best_values = []
            num_f_evals = 0

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
                num_f_evals += function.n

                best_point = (best_point[0], function(best_point[0]))

                # print(f"Time: {time:.4f}s. Best point found: {best_point}.")

                times.append(time)
                best_values.append(best_point[1])
            """ print_green(
                f"Average time: {np.mean(times):.4f} +- {np.std(times):.2f}s. Average number of evaluations: {num_f_evals / num_exp:.2f}. Average best value: {np.mean(best_values):.4f} +- {np.std(best_values):.2f}.\n"
            ) """
            opt_dict[optimizer_cls.__name__] = [
                np.mean(best_values),
                f"{num_f_evals / num_exp:.2f}",
                f"{np.mean(times):.4f}",
            ]
            if plot_figures:
                path = f"figures/{type(function).__name__}_{optimizer_cls.__name__}.svg"
                fig_gen.gen_figure(points, values, path=path)
        new_ranks = pprint_results_get_rank(opt_dict)

        all_ranks = merge_ranks(all_ranks, new_ranks)
    pprint_rank(all_ranks)
