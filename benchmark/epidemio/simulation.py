#
# Created in 2023 by Gaëtan Serré
#

import json
import numpy as np
from datetime import datetime
import os

from .model.model_comp import ModelComp
from .model.init_utils import population_state_init
from .model.scenario_utils import Scenario
from .model.score import score

DATE_FORMAT = "%Y-%m-%d"
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def date_ymd(arg):
    if not arg:
        return ""
    try:
        d = datetime.strptime(arg, DATE_FORMAT)
    except ValueError:
        raise ValueError("Argument must be a date with format {}".format(DATE_FORMAT))
    return d


class Simulation:
    def __init__(self):
        f = open(os.path.join(DIR_PATH, "parameters.json"), "r")
        json_data = json.load(f)
        self.fixed_parameters = json_data["fixed_parameters"]

        self.load_model()
        self.load_initial_population()

    def load_model(self):
        # Reading parameters from the json file and creating the model
        json_path = os.path.join(DIR_PATH, "model/epidemiological_parameters.json")

        self.model = ModelComp(
            json_file=json_path, r0=self.fixed_parameters["reproduction_number"]
        )

    def load_initial_population(self):
        # Creating the initial population state at time 0
        n_0 = np.array(
            [
                self.fixed_parameters["age_0_9"],
                self.fixed_parameters["age_10_19"],
                self.fixed_parameters["age_20_29"],
                self.fixed_parameters["age_30_39"],
                self.fixed_parameters["age_40_49"],
                self.fixed_parameters["age_50_59"],
                self.fixed_parameters["age_60_69"],
                self.fixed_parameters["age_70_79"],
                self.fixed_parameters["age_80_89"],
                self.fixed_parameters["age_90+"],
            ]
        )

        self.initial_population_state = population_state_init(
            length_init=self.model.length_segments,
            nb_age_groups=self.model.nb_age_groups,
            n_0=n_0,
        )

    def create_scenario(self, parameters):
        params = self.fixed_parameters | parameters

        cf_evolution = {
            date_ymd(params["T1"]): params["cf1"],
            date_ymd(params["T2"]): params["cf2"],
            date_ymd(params["T3"]): params["cf3"],
        }
        scenario = Scenario(
            date_ymd(params["start_date"]),
            params["simulation_duration"],
            cf_evolution,
            DATE_FORMAT,
        )

        return scenario

    def __call__(self, x):
        parameters = {"cf1": x[0], "cf2": x[1], "cf3": x[2]}
        scenario = self.create_scenario(parameters)

        # Running model
        results = self.model.run_simulation(
            scenario, self.initial_population_state, verbose=False
        )

        res = score(self.model, results[:, :, -1], scenario)

        return res


if __name__ == "__main__":
    simu = Simulation()
    print(simu([0.5, 0.7, 0.5]))
