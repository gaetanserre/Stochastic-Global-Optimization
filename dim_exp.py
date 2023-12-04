#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# benchmark functions
from benchmark.rosenbrock import Rosenbrock
from optims.extended_function import extended_function

# Optimizations algorithms
from optims.PRS import PRS
from optims.AdaLIPO_E import AdaLIPO_E
from optims.CMA_ES import CMA_ES
from optims.SBS import NMDS


def print_color(str, color):
    print(f"\033[{color}m" + str + "\033[0m")


print_yellow = lambda str: print_color(str, 93)
print_blue = lambda str: print_color(str, 94)


def cli():
    args = argparse.ArgumentParser()
    args.add_argument("--nb_exp", type=int, default=5)
    args.add_argument("--min_dim", type=int, default=3)
    args.add_argument("--max_dim", type=int, default=50)
    args.add_argument("--min_bound", type=int, default=-10)
    args.add_argument("--max_bound", type=int, default=10)
    return args.parse_args()


def create_bounds(dim, mi, mx):
    return np.array([(mi, mx) for _ in range(dim)])


def match_optim(optim_cls, bounds, num_evals):
    if optim_cls == PRS:
        return optim_cls(bounds, num_evals=num_evals)
    elif optim_cls == AdaLIPO_E:
        return optim_cls(bounds, max_evals=num_evals, window_slope=2, max_slope=300)
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
            bounds, n_particles=100, k_iter=[100_000], svgd_iter=500, lr=0.5
        )
    else:
        raise NotImplementedError(f"{optim_cls} not implemented.")


def create_figs(function_mean, function_std, path):
    plt.style.use("seaborn-v0_8")

    format = ["--", "-.", ":", "-"]

    for f in function_mean.keys():
        for i, optim in enumerate(function_mean[f].keys()):
            dims = []
            means = []
            stds = []
            for dim in function_mean[f][optim].keys():
                means.append(function_mean[f][optim][dim])
                stds.append(function_std[f][optim][dim])
                dims.append(dim)
            plt.fill_between(
                dims,
                np.abs(means) + stds,
                np.abs(means) - stds,
                alpha=0.3,
            )
            plt.plot(dims, np.abs(means), format[i], label=f"{optim}")
        plt.title(f"{f}")
        plt.xticks(dims)
        plt.xlabel("Dimension")
        plt.ylabel("$d_{min}$")
        plt.legend()
        plt.savefig(f"{f}_{path}.svg", bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":
    args = cli()

    functions = [Rosenbrock()]
    optim_cls = [CMA_ES, NMDS, AdaLIPO_E, PRS]
    optim_nb_evals = [2000, 2000, 2000, 2000]

    function_mean = {}
    function_std = {}
    for f in functions:
        print_yellow(f"Function: {f.__class__.__name__}.")
        optim_mean = {}
        optim_std = {}
        for optim, nb_evals in zip(optim_cls, optim_nb_evals):
            print_blue(f"Optimizer: {optim.__name__}.")
            dim_mean = {}
            dim_std = {}
            for dim in tqdm(range(args.min_dim, args.max_dim + 5, 5)):
                best_value = []
                for i in range(args.nb_exp):
                    f.n = 0
                    bounds = create_bounds(dim, args.min_bound, args.max_bound)
                    optimizer = match_optim(optim, bounds, nb_evals)

                    best_point, points, values = optimizer.optimize(f, verbose=False)
                    best_value.append(f(best_point[0]))
                dim_mean[dim] = np.mean(best_value)
                dim_std[dim] = np.std(best_value)
            optim_mean[optim.__name__] = dim_mean
            optim_std[optim.__name__] = dim_std
        function_mean[f.__class__.__name__] = optim_mean
        function_std[f.__class__.__name__] = optim_std

    # Export to json
    with open("dim_exp.json", "w") as f:
        json.dump(
            {
                "mean": function_mean,
                "std": function_std,
            },
            f,
            indent=4,
        )

    # create_figs(function_mean, function_std, "all")
    # remove keys from dict

    create_figs(function_mean, function_std, "all")

    """ for f in function_mean.keys():
        function_mean[f].pop("PRS")
        function_std[f].pop("PRS")
        function_mean[f].pop("AdaLIPO_E")
        function_std[f].pop("AdaLIPO_E")

    create_figs(function_mean, function_std, "CMA-ES_GO-SVGD") """
