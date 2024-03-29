#
# Created in 2023 by Gaëtan Serré
#

import matplotlib.pyplot as plt
import numpy as np


class FigGenerator:
    def __init__(self, f, X: np.ndarray):
        """
        Class to generate figures for visualizing the optimization process
        f: function optimized (Function)
        X: bounds of the parameters (numpy array)
        """

        self.f = f
        self.X = X

    def gen_figure(
        self,
        eval_points: np.array,
        eval_values: np.array,
        path: str = None,
    ):
        """
        Generates a figure
        path: path to save the figure (str) (optional)
        """

        if eval_points.shape[0] == 0:
            return

        dim = eval_points.shape[1]
        if dim == 1:
            self.gen_1D(eval_points, eval_values)
        elif dim == 2:
            self.gen_2D(eval_points, eval_values)
        else:
            raise ValueError(
                f"Cannot generate a figure for {dim}-dimensional functions"
            )

        if path is not None:
            plt.savefig(path, bbox_inches="tight")
        else:
            plt.show()
        plt.clf()

    def gen_1D(self, eval_points, eval_values):
        """
        Generates a figure for 1D functions
        """
        _ = plt.figure(figsize=(8, 8))

        x = np.linspace(self.X[0][0], self.X[0][1], 1000)
        y = np.array([self.f(xi) for xi in x])

        plt.plot(x, y)
        plt.scatter(
            eval_points,
            eval_values,
            c=eval_values,
            label="evaluations",
            cmap="viridis",
            zorder=2,
        )
        plt.colorbar(fraction=0.046, pad=0.04)
        # plt.plot(eval_points, eval_values, linewidth=0.5, color="black")
        plt.xlabel("$X$")
        plt.ylabel("$f(x)$")
        plt.legend()

    def gen_2D(self, eval_points, eval_values):
        """
        Generates a figure for 2D functions
        """

        x = np.linspace(self.X[0][0], self.X[0][1], 100)
        y = np.linspace(self.X[1][0], self.X[1][1], 100)
        x, y = np.meshgrid(x, y)
        z = np.array(
            [self.f(np.array([xi, yi])) for xi, yi in zip(np.ravel(x), np.ravel(y))]
        )
        Z = z.reshape(x.shape)

        _ = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection="3d", computed_zorder=False)

        ax.plot_surface(
            x, y, Z, cmap="coolwarm", linewidth=0, antialiased=True, zorder=4.4
        )

        cb = ax.scatter(
            eval_points[:, 0],
            eval_points[:, 1],
            eval_values,
            c=eval_values,
            label="evaluations",
            cmap="viridis",
            zorder=4.5,
        )

        # plt.colorbar(cb, fraction=0.046, pad=0.04)

        ax.set_xlabel("$x$", fontsize=15)
        ax.set_ylabel("$y$", fontsize=15)
        ax.legend(fontsize=15)
