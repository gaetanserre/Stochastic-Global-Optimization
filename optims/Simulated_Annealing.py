#
# Created in 2023 by Gaëtan Serré
#

from .__optimizer__ import Optimizer

import numpy as np
from .anneal import Annealer


class SimulatedAnnealing(Annealer):
    def __init__(self, domain, n_iter):
        self.domain = domain
        self.steps = n_iter

        self.function = None

        initial_state = np.random.uniform(self.domain[:, 0], self.domain[:, 1])

        super().__init__(initial_state)
        self.states = [initial_state]

    def move(self):
        self.state = np.random.uniform(self.domain[:, 0], self.domain[:, 1])

    def energy(self):
        return self.function(self.state)

    def optimize(self, function, verbose=False):
        self.function = function
        best_point, best_value = self.anneal()

        return (best_point, best_value), np.array(self.states), np.array(self.states)
