#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
import os
import argparse
from multiprocessing import Pool, cpu_count

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
from benchmark.goldstein_price import Goldstein_Price
from benchmark.camel import Camel
from benchmark.levy import Levy

# Optimizations algorithms
from optims.PRS import PRS
from optims.AdaLIPO_E import AdaLIPO_E
from optims.SBS import SBS
from optims.SBS_particles import SBS_particles
from optims.SBS_particles_hybrid import SBS_particles_hybrid
from optims.SBS_hybrid import SBS_hybrid
from optims.BayesOpt import BayesOpt
from optims.N_CMA_ES import CMA_ES
from optims.WOA import WOA
from optims.Langevin import Langevin
from optims.Simulated_Annealing import SimulatedAnnealing

from utils.utils import *
from utils.pretty_printer import pprint_results_get_rank, pprint_rank
from utils.mk_latex_table import mk_table


def match_optim(optim_cls, bounds, num_evals, is_sim=False):
    if optim_cls == PRS:
        return optim_cls(bounds, num_evals=num_evals)
    elif optim_cls == AdaLIPO_E:
        return optim_cls(bounds, max_evals=num_evals)
    elif optim_cls == SBS:
        return optim_cls(
            bounds,
            n_particles=500,
            k_iter=[10_000],
            svgd_iter=300,
            sigma=1 / 500**2,
            lr=0.1 if is_sim else 0.2,
        )
    elif optim_cls == SBS_particles:
        return optim_cls(
            bounds,
            n_particles=500,
            k_iter=[10_000],
            svgd_iter=300,
            sigma=lambda N: 1 / N**2,
            lr=0.1 if is_sim else 0.2,
        )
    elif optim_cls == SBS_hybrid:
        return optim_cls(
            bounds,
            n_particles=200,
            k_iter=[10_000],
            svgd_iter=1000,
            warm_start_iter=1000,
            sigma=1e-10,
            lr=0.1 if is_sim else 1e-3,
        )
    elif optim_cls == SBS_particles_hybrid:
        return optim_cls(
            bounds,
            n_particles=500,
            k_iter=[10_000],
            svgd_iter=1000,
            sigma=1e-10,
            lr=0.1 if is_sim else 1e-3,
            warm_start_iter=1000,
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
    elif optim_cls == Langevin:
        return optim_cls(bounds, n_particles=500, n_iter=150, kappa=10_000, init_lr=0.2)
    else:
        raise NotImplementedError(f"{optim_cls} not implemented.")


def cli():
    parser = argparse.ArgumentParser(description="Run the benchmark.")
    parser.add_argument(
        "--big-dimension",
        "-bd",
        action="store_true",
        help="Run the big dimensions benchmark.",
    )

    parser.add_argument(
        "--num-exp",
        "-ne",
        type=int,
        default=10,
        help="Number of experiment to run.",
    )

    parser.add_argument(
        "--latex",
        "-lt",
        action="store_true",
        help="Print the LateX table associated with the benchmark.",
    )

    return parser.parse_args()


def run_optimizer(args):
    optimizer_cls, num_eval, function, bounds, num_exp = args
    print_blue(f"Optimizer: {optimizer_cls.__name__}.")

    optimizer = match_optim(
        optimizer_cls,
        bounds,
        num_eval,
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

        times.append(time)
        best_values.append(best_point[1])

    print_green(f"{optimizer_cls.__name__} done.")

    return [
        np.mean(best_values),
        f"{num_f_evals / num_exp:.2f}",
        f"{np.mean(times):.4f}",
    ]


if __name__ == "__main__":
    args = cli()

    # Set Numpy random seed
    np.random.seed(42)

    num_exp = args.num_exp

    if args.big_dimension:
        functions = {
            "Ackley": [Ackley(), create_bounds(-32.768, 32.768, 50)],
            "Rastrigin": [Rastrigin(), create_bounds(-5.12, 5.12, 50)],
            "Rosenbrock": [Rosenbrock(), create_bounds(-3, 3, 50)],
            "Levy": [Levy(), create_bounds(-10, 10, 50)],
            "Sphere": [Sphere(), create_bounds(-10, 10, 50)],
        }

        optimizers_cls = [
            WOA,
            CMA_ES,
            Langevin,
            SBS,
            SBS_particles,
            SBS_hybrid,
            SBS_particles_hybrid,
        ]
        num_evals = [8_000_000, 8_000_000, 8_000_000, 0, 0, 0, 0]
    else:
        functions = {
            "Ackley": [Ackley(), create_bounds(-32.768, 32.768, 2)],
            "Branin": [Branin(), np.array([(-5, 10), (0, 15)])],
            "Drop Wave": [Drop_Wave(), create_bounds(-5.12, 5.12, 2)],
            "Egg Holder": [EggHolder(), create_bounds(-512, 512, 2)],
            "Goldstein Price": [Goldstein_Price(), create_bounds(-2, 2, 2)],
            "Himmelblau": [Himmelblau(), create_bounds(-4, 4, 2)],
            "Holder Table": [Holder(), create_bounds(-10, 10, 2)],
            "Rastrigin": [Rastrigin(), create_bounds(-5.12, 5.12, 2)],
            "Rosenbrock": [Rosenbrock(), create_bounds(-3, 3, 2)],
            "Camel": [Camel(), create_bounds(-3, 3, 2)],
            "Levy": [Levy(), create_bounds(-10, 10, 2)],
            "Sphere": [Sphere(), create_bounds(-10, 10, 2)],
        }

        optimizers_cls = [
            AdaLIPO_E,
            BayesOpt,
            Langevin,
            SBS,
            SBS_particles,
            SBS_hybrid,
            SBS_particles_hybrid,
            CMA_ES,
            WOA,
        ]
        num_evals = [2000, 100, 800_000, 0, 0, 0, 0, 800_000, 800_000]

    # Number of core to use
    nb_core = min(cpu_count(), len(optimizers_cls))
    print_green(f"Using {nb_core} core(s).")

    # Data to store
    all_ranks = {}
    sota_methods = {}
    proposed_methods = {}
    sbs_pf_economy = []
    sbs_pf_hybrid_economy = []

    for function_name, objects in functions.items():
        print_pink(f"Function: {function_name}.")

        function = objects[0]
        bounds = objects[1]

        dim = bounds.shape[0]
        plot_figures = False  # dim <= 2
        if plot_figures:
            from utils.fig_generator import FigGenerator

            fig_gen = FigGenerator(function, bounds)

            if not os.path.exists("figures"):
                os.makedirs("figures")

        opt_dict = {}

        args_list = [
            (optimizers_cls[i], num_evals[i], function, bounds, num_exp)
            for i in range(len(optimizers_cls))
        ]

        with Pool(nb_core) as p:
            l = p.map(run_optimizer, args_list)
        for i in range(len(optimizers_cls)):
            opt_dict[optimizers_cls[i].__name__] = l[i]

        new_ranks = pprint_results_get_rank(opt_dict)
        all_ranks = merge_ranks(all_ranks, new_ranks)

        for m_name in opt_dict.keys():
            if "SBS" in m_name:
                add_dict_entry(
                    proposed_methods, m_name, {function_name: opt_dict[m_name][0]}
                )
            else:
                add_dict_entry(
                    sota_methods, m_name, {function_name: opt_dict[m_name][0]}
                )

        if "SBS" in opt_dict and "SBS_particles" in opt_dict:
            sbs_pf_economy.append(
                100
                - float(opt_dict["SBS_particles"][1]) / float(opt_dict["SBS"][1]) * 100
            )
        if "SBS_hybrid" in opt_dict and "SBS_particles_hybrid" in opt_dict:
            sbs_pf_hybrid_economy.append(
                100
                - float(opt_dict["SBS_particles_hybrid"][1])
                / float(opt_dict["SBS_hybrid"][1])
                * 100
            )

    pprint_rank(all_ranks)

    if sbs_pf_economy != []:
        print(
            f"Average economy of SBS_particles over SBS: {np.mean(sbs_pf_economy):.2f}%."
        )
    if sbs_pf_hybrid_economy != []:
        print(
            f"Average economy of SBS_particles_hybrid over SBS_hybrid: {np.mean(sbs_pf_hybrid_economy):.2f}%."
        )

    if args.latex:
        functions_mins = {}
        for f_name in functions.keys():
            functions_mins[f_name] = functions[f_name][0].min

        mk_table(
            functions_mins,
            sota_methods,
            proposed_methods,
            all_ranks,
        )
