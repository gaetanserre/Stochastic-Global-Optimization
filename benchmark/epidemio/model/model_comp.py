"""Definition of the compartmental model."""

# from datetime import datetime, date, timedelta

from .init_utils import *


class ModelComp:
    """
    This class defines a predictive epidemiological model. This model is said to be "compartmental" because it
    classifies the entire population studied in different boxes:
    - S (Susceptible),
    - J (Non-critically infected),
    - Y (Critically Infected: This refers to those infected whose lives are threatened by the disease. They need ICU to
    have a chance of survival),
    - W (Critically ill patients who are hospitalized outside of intensive care: this refers to critically ill patients
    who will not benefit from a long stay in intensive care. They die after a short stay in intensive care or another
    care unit),
    - H (Long-stay intensive care unit: this refers to critically ill patients who are admitted to intensive care for at
    least one day. After this stay in intensive care, they may die or recover),
    - A (Recovered - the model considers a recovered individual to be immune),
    - D (Deceased).

    Each compartment is updated daily based on transition probabilities between compartments. These probabilities vary
    according to the age of the individuals, hence all compartments are divided by age group. Compartments J, Y, W and H
    are also divided into temporal sub-compartments according to the number of days since which one is in the state of
    interest. This allows the temporality of the disease to be taken into account.

    The transition probabilities are all fixed, except for the probability of being infected: this probability depends
    on the current density of infected individuals in the population (calculated from the number of individuals in J and
    Y) as well as a contact factor (or contact factor "cf"). This contact factor captures the density of contacts within
    the population under study (between 0 and 1, with 1 being the non-epidemic contact factor). The contact factor
    varies over time and is difficult to estimate. Estimating a good contact factor is a key point in the quality of the
    results.

    The model is described in the IPOL article "A Compartmental Epidemiological Model applied to the Covid-19 Epidemic"
    (Baron, Boulant, https://www.ipol.im/pub/pre/323/), itself inspired by the model developed by Samuel Alizon and the
    ETE group (https://hal.archives-ouvertes.fr/hal-02619546/document).
    """

    def __init__(self, json_file="model/epidemiological_parameters.json", r0=None):
        """
        Initialization of the object, with the age classes, the total population and the transitions from one
        compartment to another.
        Args:
            - args: parsed arguments from the user interface
            - json_file: static file containing the model epidemiological parameters.
        """

        # Reading the parameters from the json file
        lim_age_groups, length_init, model_dynamics_parameters = read_parameters(
            json_file
        )
        self.lim_age_groups = lim_age_groups
        self.nb_age_groups = len(self.lim_age_groups)
        self.parameters = model_dynamics_parameters
        self.length_segments = length_init

        # Defining the R_0
        if r0 is not None:
            self.parameters["R_0"] = r0

    def update_state(self, state, contact_factor):
        """
        Daily update of all compartments (S, J, Y, H, W, R and D) based on transition probabilities.
        Args:
        - state: current distribution of the population among compartments.
        - contact factor: current value of the contact factor.
        Returns:
        - New state of the population.
        """
        # Getting the population in each compartment
        s, j, y, h, w, r, d = self.get_all_states(state).values()

        # Population updates calculus

        # Computing the "density of contagious people" from J and Y
        # (J and Y: non critical and critical infected outside hospital).
        # Note that zeta is a vector that indicates the probability of infecting someone on day k of the infection
        # (all values are between 0 and 1, sum(zeta)=1).
        if j.shape[1] < y.shape[1]:
            contagious = np.hstack(
                (
                    np.multiply(self.parameters["zeta_j"], j),
                    np.zeros((j.shape[0], y.shape[1] - j.shape[1])),
                )
            ) + np.multiply(self.parameters["zeta_y"], y)
        else:
            contagious = np.hstack(
                (
                    np.multiply(self.parameters["zeta_y"], y),
                    np.zeros((y.shape[0], j.shape[1] - y.shape[1])),
                )
            ) + np.multiply(self.parameters["zeta_j"], j)
        density_contagious = np.sum(
            np.multiply(contact_factor, np.transpose(contagious))
        )

        if density_contagious == 0:
            # If there is no contagious people anymore
            force_infection = 0
        else:
            # Force_infection is the probability for a susceptible individual to be infected.
            force_infection = density_contagious / (
                np.sum(s) / np.multiply(contact_factor, self.parameters["R_0"])
                + density_contagious
            )

        # New infected, either in Y or in J
        si = np.multiply(s, force_infection)

        # Critical (Y) or non critical (J) newly infected
        iy = np.multiply(si, self.parameters["theta"])
        ij = si - iy

        # Evolution of compartment Y
        y_out = np.multiply(self.parameters["eta"], y)
        yy = y - y_out
        yh = np.multiply(self.parameters["psi"].reshape(-1, 1), y_out)
        yw = y_out - yh
        y_n = np.roll(yy, shift=1, axis=1)
        y_n[:, 0] = iy

        # Evolution of compartment J
        jj = j
        jr = j[:, -1]
        j_n = np.roll(jj, shift=1, axis=1)
        j_n[:, 0] = ij

        # Evolution of compartment H
        h_out = np.multiply(h, self.parameters["rho"])
        hh = h - h_out
        hd = np.multiply(np.sum(h_out, axis=1), self.parameters["mu"])
        hr = np.sum(h_out, axis=1) - hd
        h_n = np.roll(hh, shift=1, axis=1)
        h_n[:, 0] = np.sum(yh, axis=1)

        # Evolution of compartment W
        w_out = np.multiply(w, self.parameters["v"])
        ww = w - w_out
        wd = np.sum(w_out, axis=1)
        w_n = np.roll(ww, shift=1, axis=1)
        w_n[:, 0] = np.sum(yw, axis=1)

        # Evolution of compartments S, R and D
        s_n = s - si
        r_n = r + jr + hr
        d_n = d + hd + wd

        return np.stack(
            [
                s_n,
                *np.transpose(j_n),
                *np.transpose(y_n),
                *np.transpose(h_n),
                *np.transpose(w_n),
                r_n,
                d_n,
            ]
        )

    def run_simulation(self, scenario, initial_population_state, verbose=False):
        """
        Runs the simulation according to the chosen scenario.
        Args:
        - scenario: chosen scenario
        - verbose=False: printing comments on/off
        Returns:
        - sol_tot: concatenation of all the daily states of the population during the simulation.
        """

        # Initialization
        political_measure_dates = np.array(scenario.contact_factor_evolution["dates"])
        contact_factor_list = scenario.contact_factor_evolution["contact_factor"]

        # Array with all the simulations of the scenarii
        sol_tot = np.zeros(
            (
                scenario.t_end + 1,
                self.length_segments["length_g"]
                + self.length_segments["length_h"]
                + self.length_segments["length_r"]
                + self.length_segments["length_u"]
                + 3,
                self.nb_age_groups,
            )
        )

        # Initial conditions
        sol_tot[0] = np.vstack(
            [
                initial_population_state["S_0"],
                np.transpose(initial_population_state["J_0"]),
                np.transpose(initial_population_state["Y_0"]),
                np.transpose(initial_population_state["H_0"]),
                np.transpose(initial_population_state["W_0"]),
                initial_population_state["R_0"],
                initial_population_state["D_0"],
            ]
        )
        t = 0

        # Starting the simulation
        while t < scenario.t_end:
            if verbose:
                print("Time %d" % (t + 1))
            # Get right contact factor
            contact_factor = contact_factor_list[
                np.sum(t >= political_measure_dates) - 1
            ]
            # Updating states according to system of recurrence relations
            sol_tot[t + 1] = self.update_state(sol_tot[t], contact_factor)
            t += 1

        if verbose:
            print("Simulation complete.")

        # Adding a column corresponding to the sum of all age groups
        sol_tot_allage = np.expand_dims(np.sum(sol_tot, axis=-1), axis=-1)
        sol_tot = np.concatenate([sol_tot, sol_tot_allage], axis=-1)

        return sol_tot

    def get_all_states(self, state):
        "Returns each compartment state from the solution array."

        # Compartment order in the array
        state_order = [
            "S",
            "J",
            *["J_"] * (self.length_segments["length_g"] - 1),
            "Y",
            *["Y_"] * (self.length_segments["length_h"] - 1),
            "H",
            *["H_"] * (self.length_segments["length_r"] - 1),
            "W",
            *["W_"] * (self.length_segments["length_u"] - 1),
            "R",
            "D",
        ]
        idx_j = state_order.index("J")
        idx_y = state_order.index("Y")
        idx_h = state_order.index("H")
        idx_w = state_order.index("W")
        return {
            "S": state[state_order.index("S")],
            "J": np.transpose(state[idx_j : idx_j + self.length_segments["length_g"]]),
            "Y": np.transpose(state[idx_y : idx_y + self.length_segments["length_h"]]),
            "H": np.transpose(state[idx_h : idx_h + self.length_segments["length_r"]]),
            "W": np.transpose(state[idx_w : idx_w + self.length_segments["length_u"]]),
            "R": state[state_order.index("R")],
            "D": state[state_order.index("D")],
        }
