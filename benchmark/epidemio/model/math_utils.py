"""Contains all auxiliary mathematical functions used for probabilities calculation."""

import numpy as np
import math


def weibull_cdf(x, k, lbd):
    """Cumulative distribution function of Weibull(k, lambda)"""
    return 1 - math.exp(-(x / lbd) ** k)


def exp_cdf(x, lbd):
    """Cumulative distribution function of Exp(lambda)"""
    return 1 - math.exp(-x * lbd)


def truncate(params, dist="Weibull"):
    """Computes the truncation and normalization of a distribution."""
    if dist == "Weibull":
        cdf = weibull_cdf
    elif dist == "Exp":
        cdf = exp_cdf
    else:
        raise ValueError("Unknown distribution")
    F, i = 0., 0
    my_list = []
    while F < .99:
        i += 1
        F = cdf(i, *params)
        my_list.append(F)
    return np.array(my_list) / my_list[-1]


def diff(array):
    """Returns the differences array[k+1]-array[k]. The input array is the cdf values of integers."""
    array_bis = np.zeros(array.shape)
    array_bis[1:] = array[:-1]
    return array - array_bis


def conditioned(array):
    """Returns the conditioned probabilities. The input array is the cdf values of integers."""
    array_bis = np.zeros(array.shape)
    array_bis[1:] = array[:-1]
    return (array - array_bis) / (1. - array_bis)
