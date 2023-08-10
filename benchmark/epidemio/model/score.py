#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
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
    cf_values = np.array(scenario.contact_factor_evolution["contact_factor"])

    # Score = nb_dead * max(1, log(1 / c1 * c2 * c3))
    score = d[-1] * np.maximum(1, np.log(1 / (cf_values.prod() + 1e-10)))

    return score
