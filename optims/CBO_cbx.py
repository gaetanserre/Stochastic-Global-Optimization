#
# Created in 2024 by Gaëtan Serré
#

from .__optimizer__ import Optimizer

import numpy as np
from cbx.dynamics import CBO as CBO_opt


class post_process:
    def __init__(self, domain):
        self.domain = domain

    def __call__(self, dyn):
        np.nan_to_num(dyn.x, copy=False, nan=1e8)
        dyn.x = np.clip(dyn.x, self.domain[:, 0], self.domain[:, 1])


class CBO_cbx(Optimizer):
    def __init__(self, domain, n_iter, n_particles):
        self.domain = domain
        self.n_iter = n_iter
        self.n_particles = n_particles

    def optimize(self, function, verbose=False):
        optimizer = CBO_opt(
            f=function,
            N=self.n_particles,
            d=len(self.domain),
            max_it=self.n_iter,
            post_process=post_process(self.domain),
            verbosity=int(verbose),
        )

        x = optimizer.optimize()
        return (x[0], function(x[0])), None, None
