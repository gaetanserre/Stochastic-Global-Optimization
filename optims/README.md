## Global optimization algorithms

This repository implements several global optimization algorithms. The algorithms are implemented in the `optims` package. The `benchmark` folder contains usual test functions for global optimization. Run `main.py` to run the algorithms on the benchmark functions.

### Algorithms

- Pure Random Search
- CMA-ES [Hansen, 2023](https://inria.hal.science/hal-01297037/file/tutorial-2023-02.pdf)
- AdaLIPO-E [Malherbe and Vayatis, 2017](https://arxiv.org/pdf/1812.03457.pdf) & [Beja-Battais et al., 2023](https://hal-universite-paris-saclay.archives-ouvertes.fr/hal-04069150/document)
- GO-SVGD, a global optimization algorithm based on the *nassant minimas distribution* of [Luo, 2019](https://arxiv.org/abs/1812.03457) and uses the *Stein Variational Gradient Descent* algorithm of [Liu and Wang, 2016](https://arxiv.org/abs/1608.04471) to approximate the distribution through variational inference.

### TODO

- [x] Fix histogram for high dimensions
- [ ] Add more examples
- [ ] Approximation Laplace integral for high dimension (GO-SVGD)