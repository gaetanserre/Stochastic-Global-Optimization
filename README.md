## Stochastic Global optimization algorithms

This repository implements several global optimization algorithms including SBS. The algorithms are implemented in the `optims` package. The `benchmark` folder contains usual test functions for global optimization. Run `main.py` to run the algorithms on the benchmark functions.

### Algorithms

- [SBS](https://hal.science/hal-04442217)
- Pure Random Search
- CMA-ES [Hansen, 2023](https://inria.hal.science/hal-01297037/file/tutorial-2023-02.pdf)
- AdaLIPO-E [Malherbe and Vayatis, 2017](https://arxiv.org/pdf/1812.03457.pdf) & [Beja-Battais et al., 2023](https://hal-universite-paris-saclay.archives-ouvertes.fr/hal-04069150/document)
- Whale Optimization Algorithm [Mirjalili and Lewis, 2016](https://www.sciencedirect.com/science/article/pii/S0965997816300163)
- Bayesian Optimization [Brochu et al., 2016](https://arxiv.org/pdf/1012.2599.pdf)

### Credits

[Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/optimization.html)