#
# Created in 2023 by Gaëtan Serré
#

import jax
import jax.numpy as jnp
from .__function__ import Function


class Michalewicz(Function):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    @staticmethod
    def call(x: jnp.ndarray) -> float:
        return -jnp.sum(
            jnp.array(
                [
                    jnp.sin(x[i]) * jnp.sin((i + 1) * x[i] ** 2 / jnp.pi) ** (2 * 10)
                    for i in range(x.shape[0])
                ]
            )
        )

    def __call__(self, x: jnp.ndarray) -> float:
        self.n += 1
        jit = jax.jit(self.call)
        return jit(x)
