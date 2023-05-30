"""Creation of graphical outputs."""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.dates import drange, DateFormatter, DayLocator, MonthLocator
from datetime import timedelta

FIGSIZE = (19, 7)


def format_plot(fig, legend=True):
    """
    Formatting the figure fig for readability, with a legend and the right date format.
    """
    axis = fig.get_axes()[0]
    if legend:
        plt.legend()
    fig.autofmt_xdate()
    months = MonthLocator()
    days = DayLocator()
    axis.xaxis.set_major_locator(months)
    axis.xaxis.set_major_formatter(DateFormatter('%B %y'))
    axis.xaxis.set_minor_locator(days)


def ipol_plot(model, results, scenario=None):
    """Plots all the categories evolution over time from the results."""
    # Transforming results to the right format
    s, j, y, h, w, r, d = model.get_all_states(np.transpose(results)).values()
    if scenario is None:
        t = np.arange(len(s))
    else:
        dstart = scenario.date_start
        dend = dstart + timedelta(days=scenario.t_end + 1)
        t = drange(dstart, dend, timedelta(days=1))

        # cf
        fig_cf = plt.figure(figsize=FIGSIZE)
        cf_evolution = scenario.contact_factor_evolution['contact_factor']
        dates = scenario.contact_factor_evolution['dates']
        cf = [cf_evolution[np.sum(t >= dates) - 1] for t in range(scenario.t_end + 1)]
        plt.scatter(t, cf, figure=fig_cf, marker='+', label='Contact factor evolution')
        format_plot(fig_cf, legend=False)
        plt.ylim((0, 1.05))
        plt.title('Contact factor evolution')
        plt.ylabel('Contact factor')
        plt.savefig("sim_results_cf.png")

    # SR
    fig_SR = plt.figure(figsize=FIGSIZE)
    plt.plot(t, s, figure=fig_SR, label='S')
    plt.plot(t, r, figure=fig_SR, label='R')
    format_plot(fig_SR)
    plt.title('Susceptible (S) and recovered (R) individuals')
    plt.ylabel('Number of individuals')
    plt.savefig("sim_results_SR.png")

    # J
    fig_J = plt.figure(figsize=FIGSIZE)
    plt.plot(t, j.sum(axis=1), figure=fig_J, label='J')
    format_plot(fig_J)
    plt.title('Non critical infections (J)')
    plt.ylabel('Number of individuals')
    plt.savefig("sim_results_J.png")

    # Y
    fig_Y = plt.figure(figsize=FIGSIZE)
    plt.plot(t, y.sum(axis=1), figure=fig_Y, label='Y')
    format_plot(fig_Y)
    plt.title('Critical infections (Y)')
    plt.ylabel('Number of individuals')
    plt.savefig("sim_results_Y.png")

    # HW
    fig_HW = plt.figure(figsize=FIGSIZE)
    plt.plot(t, h.sum(axis=1), figure=fig_HW, label='H')
    plt.plot(t, w.sum(axis=1), figure=fig_HW, label='W')
    format_plot(fig_HW)
    plt.title('Covid-19 hospitalizations in long-stay ICU (H) or outside ICU (W)')
    plt.ylabel('Number of individuals')
    plt.savefig("sim_results_HW.png")

    # D
    fig_D = plt.figure(figsize=FIGSIZE)
    plt.plot(t, d, figure=fig_D, label='D')
    format_plot(fig_D)
    plt.title('Number of deaths (D)')
    plt.ylabel('Number of individuals')
    plt.savefig("sim_results_D.png")
