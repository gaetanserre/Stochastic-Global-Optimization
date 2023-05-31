#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from scipy.integrate import cumulative_trapezoid as integrate
from matplotlib.dates import drange
from datetime import timedelta


def score(model, results, scenario: None):
    s, j, y, h, w, r, d = model.get_all_states(np.transpose(results)).values()

    if scenario is None:
        t = np.arange(len(s))
    else:
        dstart = scenario.date_start
        dend = dstart + timedelta(days=scenario.t_end + 1)
        t = drange(dstart, dend, timedelta(days=1))

    # Compute the score
    h = h.sum(axis=1)
    h = h

    w = w.sum(axis=1)
    w = w

    cf_values = np.array(scenario.contact_factor_evolution["contact_factor"])

    score = d[-1] / (
        cf_values.prod() + 1e-10
    )  # integrate(h, t)[-1] + integrate(w, t)[-1] / (cf_values.prod() + 1e-10)

    return -score
