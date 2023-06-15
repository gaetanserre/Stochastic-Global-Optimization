#
# Created in 2023 by Gaëtan Serré
#

import argparse
import numpy as np
import subprocess
import os
import json
import time

from optims.extended_function import extended_function

from optims.PRS import PRS
from optims.AdaLIPO_E import AdaLIPO_E
from optims.CMA_ES import CMA_ES
from optims.GO_SVGD import GO_SVGD

from epidemiology.simulation import Simulation

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# IPOL utils functions


def return_error(error):
    with open("demo_failure.txt", "w") as f:
        f.write(error)
    exit(0)


# Parse the bounds
def parse_bounds(bounds):
    bounds = []
    for b in args.bounds.split(" "):
        try:
            bounds.append(float(b))
        except:
            return_error("Bounds must be float.")
    return np.array(bounds).reshape(-1, 2)


# Use the OCaml parser to parse the function
def parse_function(function):
    if (
        subprocess.run(
            [
                os.path.join(DIR_PATH, "numpy_parser", "numpy_parser"),
                os.path.join(DIR_PATH, "numpy_parser", "numpy_primitives.txt"),
                function,
            ]
        ).returncode
        != 0
    ):
        return_error(f"Function expression '{args.function}' is not valid.")
    return lambda x: eval(function)


def build_ifr(death_param, nb_age_group, d=2.2):
    ifr = [0] * nb_age_group
    for i in range(nb_age_group - 1, -1, -1):
        ifr[i] = death_param / d ** ((nb_age_group - 1) - i)
    return ifr


def time_it(function, args={}):
    start = time.time()
    ret = function(**args)
    end = time.time()
    return ret, end - start


def cli():
    argparser = argparse.ArgumentParser(description="Run the experiments.")
    argparser.add_argument(
        "-e", "--nb_eval", type=int, default=1000, help="Number of evaluations."
    )
    argparser.add_argument(
        "-x", "--nb_exp", type=int, default=5, help="Number of experiments."
    )
    argparser.add_argument(
        "-f", "--function", type=str, default="", help="Function to optimize."
    )
    argparser.add_argument("-b", "--bounds", type=str, default="", help="Bounds.")
    argparser.add_argument(
        "-r",
        "--reproduction",
        type=float,
        default=2.46,
        help="Reproduction number for epidemiology simulation.",
    )
    argparser.add_argument(
        "-d",
        "--death_rate",
        type=float,
        default=0.008,
        help="Death rate for epidemiology simulation.",
    )

    return argparser.parse_args()


def run_exps(
    optimizers_cls, nb_exp, function, bounds, nb_eval, is_simu=False, plot_figures=False
):
    if plot_figures:
        from fig_generator import FigGenerator

        fig_gen = FigGenerator(function, bounds)

        if not os.path.exists("figures"):
            os.makedirs("figures")

    for optimizer_cls in optimizers_cls:
        print(f"Optimizer: {optimizer_cls.__name__}.")
        if optimizer_cls == PRS:
            optimizer = optimizer_cls(bounds, num_evals=nb_eval)
        elif optimizer_cls == AdaLIPO_E:
            optimizer = optimizer_cls(bounds, max_evals=nb_eval)
        elif optimizer_cls == CMA_ES:
            m_0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
            optimizer = optimizer_cls(
                bounds,
                m_0,
                num_generations=nb_eval // 5,
                lambda_=5,
                cov_method="full",
            )
        elif optimizer_cls == GO_SVGD:
            optimizer = optimizer_cls(
                bounds,
                n_particles=10,
                k_iter=[100_000],
                svgd_iter=10 if is_simu else 1000,
                lr=0.1 if is_simu else 0.1,
            )
        else:
            raise return_error(f"{optimizer_cls} not implemented.")

        times = []
        best_values = []
        points = None
        values = None
        old_best = None
        for _ in range(nb_exp):
            ret, time = time_it(
                optimizer.optimize,
                {"function": extended_function(function, bounds), "verbose": False},
            )
            best_point, points_, values_ = ret
            best_point = (best_point[0], function(best_point[0]))
            if old_best == None or best_point[1] > old_best:
                points = points_
                values = values_
                old_best = best_point[1]

            print(f"Time: {time:.4f}s. Best point found: {best_point}.")

            times.append(time)
            best_values.append(best_point[1])
        print(
            f"Average time: {np.mean(times):.4f} +- {np.std(times):.2f}s. Average best value: {np.mean(best_values):.4f} +- {np.std(best_values):.2f}.\n"
        )

        if plot_figures:
            path = f"figures/fun_{optimizer_cls.__name__}.png"
            fig_gen.gen_figure(points, values, optimizer_cls.__name__, path=path)


if __name__ == "__main__":
    args = cli()

    optimizers_cls = [PRS, AdaLIPO_E, CMA_ES, GO_SVGD]

    if args.function == "":
        # Experimental design optimization

        # Set the reproduction number
        params = json.load(
            open(os.path.join(DIR_PATH, "epidemiology", "parameters.json"), "r")
        )
        params["reproduction_number"] = args.reproduction

        json.dump(
            params,
            open(os.path.join(DIR_PATH, "epidemiology", "parameters.json"), "w"),
            indent=4,
        )

        disease_param = json.load(
            open(
                os.path.join(
                    DIR_PATH, "epidemiology", "model", "epidemiological_parameters.json"
                ),
                "r",
            )
        )

        # Set the mortality
        ifr = build_ifr(args.death_rate, 10)

        disease_param["transition_data"]["ifr"] = ifr

        json.dump(
            disease_param,
            open(
                os.path.join(
                    DIR_PATH, "epidemiology", "model", "epidemiological_parameters.json"
                ),
                "w",
            ),
            indent=4,
        )

        simulation = Simulation()
        bounds = np.array([(0, 1), (0, 1), (0, 1)])

        run_exps(
            optimizers_cls,
            1,
            simulation,
            bounds,
            min(args.nb_eval, 250),
            is_simu=True,
            plot_figures=False,
        )

    else:
        # Mathematical function optimization
        f = parse_function(args.function)
        bounds = parse_bounds(args.bounds)
        dim = bounds.shape[0]

        run_exps(
            optimizers_cls, args.nb_exp, f, bounds, args.nb_eval, plot_figures=dim <= 2
        )
