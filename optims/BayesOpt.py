#
# Created in 2023 by Gaëtan Serré
#

from .__optimizer__ import Optimizer

import numpy as np
from bayes_opt import BayesianOptimization


class BayesOpt(Optimizer):
    def __init__(self, domain, n_iter):
        self.domain = domain
        self.n_iter = n_iter

        self.create_optimizer = lambda function: BayesianOptimization(
            f=self.transform_function(function),
            pbounds=self.transform_domain(domain),
            verbose=0,
            allow_duplicate_points=True,
        )

    @staticmethod
    def transform_domain(domain):
        p_bounds = {}
        for i, bounds in enumerate(domain):
            p_bounds[f"x{i}"] = bounds
        return p_bounds

    @staticmethod
    def transform_function(function):
        def intermediate_fun(**params):
            return -function(np.array(list(params.values())))

        return intermediate_fun

    @staticmethod
    def dict_values_to_array(params):
        point = [0] * len(params)
        for i, v in enumerate(params):
            point[i] = v
        return np.array(point)

    def optimize(self, function, verbose=False):
        optimizer = self.create_optimizer(function)
        optimizer.maximize(n_iter=self.n_iter)

        # Recover best point/value
        best_value = -optimizer.max["target"]
        best_point = self.dict_values_to_array(optimizer.max["params"].values())

        # Recover all points/values
        o_res = optimizer.res

        points = [0] * len(o_res)
        values = [0] * len(o_res)

        for i, res in enumerate(o_res):
            values[i] = -res["target"]
            points[i] = self.dict_values_to_array(res["params"].values())

        return (best_point, best_value), np.array(points), np.array(values)
