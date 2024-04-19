#
# Created in 2024 by Gaëtan Serré
#

from .__optimizer__ import Optimizer

from .SBS_particles import SBS_particles


class SBS_particles_hybrid(SBS_particles):
    def __init__(
        self,
        domain,
        n_particles,
        k_iter,
        svgd_iter,
        warm_start_iter,
        sigma=1e-10,
        distance_q=0.5,  # 0
        value_q=0.3,  # 0
        lr=0.2,
        adam=True,
    ):
        if type(sigma) == float:
            v = sigma
            sigma = lambda N: v

        super().__init__(
            domain,
            n_particles,
            k_iter,
            svgd_iter,
            sigma=sigma,
            distance_q=distance_q,
            value_q=value_q,
            lr=lr,
            adam=adam,
            warm_start_iter=warm_start_iter,
        )
