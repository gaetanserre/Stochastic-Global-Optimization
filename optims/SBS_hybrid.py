#
# Created in 2024 by Gaëtan Serré
#

from .SBS import SBS


class SBS_hybrid(SBS):
    def __init__(
        self,
        domain,
        n_particles,
        k_iter,
        svgd_iter,
        warm_start_iter,
        sigma=1e-10,
        lr=0.2,
    ):
        super().__init__(
            domain,
            n_particles,
            k_iter,
            svgd_iter,
            sigma=sigma,
            lr=lr,
            warm_start_iter=warm_start_iter,
        )
