"""Contains all functions needed for initialization."""

import json

from .math_utils import *


def probabilities_init(eta_weibull_k,
                       eta_weibull_lbd,
                       rho_exp_lbd,
                       zeta_weibull_k,
                       zeta_weibull_lbd,
                       v_exp_lbd,
                       mu,
                       icu,
                       d,
                       ifr):
    """Initializes branching probabilities, interval distributions and transmission parameters.
    Args:
        :param r0: initial basic reproduction number,
        :param eta_weibull_k: k parameter of the Weibull distribution of eta,
        :param eta_weibull_lbd: lambda parameter of the Weibull distribution of eta,
        :param rho_exp_lbd: lambda parameter of the Weibull distribution of eta,
        :param zeta_weibull_k: lambda parameter of the Weibull distribution of eta,
        :param zeta_weibull_lbd: lambda parameter of the Weibull distribution of eta,
        :param v_exp_lbd: lambda parameter of the Weibull distribution of eta,
        :param mu: lambda parameter of the Weibull distribution of eta,
        :param icu: lambda parameter of the Weibull distribution of eta,
        :param d: lambda parameter of the Weibull distribution of eta,
        :param ifr: lambda parameter of the Weibull distribution of eta,
    :returns a dictionary containing g, h, r and u (which are respectively the number of subcompartments in J, Y, H and
    W) and an other dictionary with all the probabilities needed to run the model.
    """

    # Interval distributions
    zeta = diff(truncate((zeta_weibull_k, zeta_weibull_lbd), dist="Weibull"))
    eta = conditioned(truncate([eta_weibull_k, eta_weibull_lbd], dist="Weibull"))
    rho = conditioned(truncate([rho_exp_lbd], dist="Exp"))
    v = conditioned(truncate([v_exp_lbd], dist="Exp"))

    # Lengths of compartments
    g, h, r, u = zeta.shape[0], eta.shape[0], rho.shape[0], v.shape[0]

    # Transmission probabilities for Y compartment
    zeta_y = np.zeros(h)
    zeta_y[:min(g, h)] = zeta[:min(g, h)]

    # Branching probabilities
    psi = np.clip(1 / (1 - mu + (d / icu)), 0., 1.)
    theta = np.clip(ifr / (1 - (1 - mu) * psi), 0., 1.)

    return {'length_g': g,
            'length_h': h,
            'length_r': r,
            'length_u': u}, \
           {'zeta': zeta,
            'zeta_j': zeta,
            'zeta_y': zeta_y,
            'eta': eta,
            'rho': rho,
            'v': v,
            'theta': theta,
            'psi': psi,
            'mu': mu}


def read_parameters(file="model/epidemiological_parameters.json"):
    """
    Reads the json file and extracts all static parameters from it.
    Args:
        :param file: path to the file containing the parameters.
    :returns
        - lim_age_groups: lower bound of each age group.
        - length_init: number of time sub-compartments of zeta, eta, rho and v (g, h, r and u respectively).
        - model_dynamics_parameters: all probabilities needed to initialize the model.
        - simulation parameters: dictionary with 'date_format', 'date_start' (day corresponding to both id=0
        and the day of the first case in France) and 'default contact factor' (contact factor at the very
        beginning of the epidemics, conventionally 1).
    """

    # Reading data
    with open(file, "r") as f:
        data = f.read()
    all_parameters = json.loads(data)

    # Initializing all probabilities
    for proba_cat in ['mu', 'icu', 'd', 'ifr']:
        all_parameters['transition_data'][proba_cat] = np.array(all_parameters['transition_data'][proba_cat])
    length_init, model_dynamics_parameters = probabilities_init(**all_parameters['transition_data'])

    # Population parameters
    lim_age_groups = np.array(all_parameters['age_groups'])

    return lim_age_groups, length_init, model_dynamics_parameters


def population_state_init(length_init, nb_age_groups, n_0):
    """
    Initialization of the number of individuals in each state at the very beginning of the epidemics. There is only one
    non critical infected. All other individuals are susceptible.
    Args:
        :param length_init: dictionary containing g, h, r and u which are respectively the number of subcompartments in
        J, Y, H and W.
        :param nb_age_groups: number of age groups (integer - does not include the total of all age groups).
        :param n_0: Total studied population at time 0, excluding people in nursing homes.
    :returns a dictionary containing the number of individual per age group in each compartment (and the total N_0) at
    time 0.
    """

    # Initializing J
    j00 = 1
    J_0 = np.zeros((nb_age_groups, length_init['length_g']))
    J_0[:, 0] = j00 / nb_age_groups

    # Initializing all other compartments (except S) to zero
    Y_0 = np.zeros((nb_age_groups, length_init['length_h']))
    H_0 = np.zeros((nb_age_groups, length_init['length_r']))
    W_0 = np.zeros((nb_age_groups, length_init['length_u']))
    Recovered_0 = np.zeros(nb_age_groups)
    D_0 = np.zeros(nb_age_groups)

    # Computing S (Susceptible) compartment
    S_0 = n_0 - np.sum(J_0, axis=1) - np.sum(Y_0, axis=1)

    return {'N_0': n_0,  # Total number of individuals at time 0
            'R_0': Recovered_0,  # Total number of immune at time 0
            'S_0': S_0,  # Total number of susceptible at time 0
            'D_0': D_0,  # Total number of deaths at time 0
            'J_0': J_0,  # Total number of non critical infectious at time 0
            'Y_0': Y_0,  # Total number of Critical infectious at time 0
            'H_0': H_0,  # Total number of Long-stay ICU hospitalized at time 0
            'W_0': W_0,  # Other critical hospitalized patients at time 0
            }
