#
# Created in 2024 by Gaëtan Serré
#

import json
import os
import numpy as np


def build_ifr(death_param, nb_age_group, d=2.2):
    ifr = [0] * nb_age_group
    for i in range(nb_age_group - 1, -1, -1):
        ifr[i] = death_param / d ** ((nb_age_group - 1) - i)
    return ifr


DIR_PATH = "benchmark"


def init_epidemiology():
    params = json.load(
        open(os.path.join(DIR_PATH, "epidemiology", "parameters.json"), "r")
    )
    params["reproduction_number"] = 2.46

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
    ifr = build_ifr(0.083, 10)

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
