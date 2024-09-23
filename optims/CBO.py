#
# Created in 2024 by Gaëtan Serré
#

from .__optimizer__ import Optimizer

import sys

import numpy as np
from cbx.dynamics import CBO as CBO_opt


class post_process:
    def __init__(self, domain):
        self.domain = domain

    def __call__(self, dyn):
        np.nan_to_num(dyn.x, copy=False, nan=1e8)
        dyn.x = np.clip(dyn.x, self.domain[:, 0], self.domain[:, 1])


class CBO(Optimizer):
    def __init__(self, domain, n_iter, n_particles):
        self.domain = domain
        self.n_iter = n_iter
        self.n_particles = n_particles

    def transform_function(self, function):
        def intermediate_fun(x):
            for i in range(len(x)):
                if x[i] < self.domain[i][0] or x[i] > self.domain[i][1]:
                    return sys.float_info.max
                else:
                    return function(x)

        return intermediate_fun

    def optimize(self, function, verbose=False):
        optimizer = CBO_opt(
            f=self.transform_function(function),
            N=self.n_particles,
            d=len(self.domain),
            max_it=self.n_iter,
            post_process=post_process(self.domain),
            verbosity=int(verbose),
        )

        x = optimizer.optimize()
        return (x[0], function(x[0])), None, None
