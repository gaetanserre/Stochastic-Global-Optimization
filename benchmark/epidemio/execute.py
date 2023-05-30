#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import numpy as np
from datetime import datetime

from model.model_comp import ModelComp
from model.init_utils import population_state_init
from model.scenario_utils import Scenario
from model.plot_utils import ipol_plot

MIN_0 = 0.0
MAX_1 = 1.0
DATE_FORMAT = "%Y-%m-%d"


def float_range_0_1_type(arg):
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < MIN_0 or f > MAX_1:
        raise argparse.ArgumentTypeError(
            "Argument must be <= " + str(MAX_1) + "and >= " + str(MIN_0)
        )
    return f


def date_ymd(arg):
    if not arg:
        return ""
    try:
        d = datetime.strptime(arg, DATE_FORMAT)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Argument must be a date with format {}".format(DATE_FORMAT)
        )
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Alizon and Al. Model")

    # Transmission parameters
    parser.add_argument(
        "r0", help="Basic reproduction number", type=float, default=2.49
    )

    # Simulation
    parser.add_argument(
        "date_start",
        help="Date - simulation start",
        type=date_ymd,
        default="2020-01-03",
    )
    parser.add_argument("t_end", help="Simulation duration", type=int, default=196)

    # Contact factor
    parser.add_argument(
        "cf_d1", help="Contact factor - date 1", type=date_ymd, default="2020-03-17"
    )
    parser.add_argument(
        "cf_v1", help="Contact factor 1", type=float_range_0_1_type, default=0.6
    )
    parser.add_argument(
        "cf_d2", help="Contact factor - date 1", type=date_ymd, default="2020-05-17"
    )
    parser.add_argument(
        "cf_v2", help="Contact factor 1", type=float_range_0_1_type, default=0.7
    )
    parser.add_argument(
        "cf_d3", help="Contact factor - date 1", type=date_ymd, default="2020-10-31"
    )
    parser.add_argument(
        "cf_v3", help="Contact factor 1", type=float_range_0_1_type, default=0.3
    )

    # Total number of individuals at time 0
    parser.add_argument(
        "N_0_0",
        help="Total number of individuals at time 0 in age class [0, 9]",
        type=int,
        default=7755755,
    )
    parser.add_argument(
        "N_0_1",
        help="Total number of individuals at time 0 in age class [10, 19]",
        type=int,
        default=8328988,
    )
    parser.add_argument(
        "N_0_2",
        help="Total number of individuals at time 0 in age class [20, 29]",
        type=int,
        default=7470908,
    )
    parser.add_argument(
        "N_0_3",
        help="Total number of individuals at time 0 in age class [30, 39]",
        type=int,
        default=8288257,
    )
    parser.add_argument(
        "N_0_4",
        help="Total number of individuals at time 0 in age class [40, 49]",
        type=int,
        default=8584449,
    )
    parser.add_argument(
        "N_0_5",
        help="Total number of individuals at time 0 in age class [50, 59]",
        type=int,
        default=8785106,
    )
    parser.add_argument(
        "N_0_6",
        help="Total number of individuals at time 0 in age class [60, 69]",
        type=int,
        default=7999606,
    )
    parser.add_argument(
        "N_0_7",
        help="Total number of individuals at time 0 in age class [70, 79]",
        type=int,
        default=5693660,
    )
    parser.add_argument(
        "N_0_8",
        help="Total number of individuals at time 0 in age class [80, 89]",
        type=int,
        default=2864543,
    )
    parser.add_argument(
        "N_0_9",
        help="Total number of individuals at time 0 in age class 90+",
        type=int,
        default=562431,
    )

    parser.add_argument("bin", help="Bin path for IPOL demo", type=str, default=None)

    args = parser.parse_args()

    # Creating the scenario
    cf_evolution = {
        args.cf_d1: args.cf_v1,
        args.cf_d2: args.cf_v2,
        args.cf_d3: args.cf_v3,
    }
    scenario = Scenario(args.date_start, args.t_end, cf_evolution, DATE_FORMAT)

    # Reading parameters from the json file and creating the model
    if args.bin:
        json_path = "{}/model/epidemiological_parameters.json".format(args.bin)
    else:
        json_path = "model/epidemiological_parameters.json"
    model = ModelComp(json_file=json_path, r0=args.r0)

    # Creating the initial population state at time 0
    n_0 = np.array(
        [
            args.N_0_0,
            args.N_0_1,
            args.N_0_2,
            args.N_0_3,
            args.N_0_4,
            args.N_0_5,
            args.N_0_6,
            args.N_0_7,
            args.N_0_8,
            args.N_0_9,
        ]
    )
    initial_population_state = population_state_init(
        length_init=model.length_segments, nb_age_groups=model.nb_age_groups, n_0=n_0
    )

    # Running model
    results = model.run_simulation(scenario, initial_population_state, verbose=False)

    # Plots
    ipol_plot(model, results[:, :, -1], scenario)
