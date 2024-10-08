#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .utils import *

# dict : Method name -> Function name -> score


def sort_methods(methods, all_ranks):
    key = lambda name: -np.mean(all_ranks[name])
    methods.sort(key=key)
    return methods


def transform_power_of_ten(v):
    n = 0
    while np.abs(v * 10**n) <= 1:
        n += 1
    return f"{int(v * 10**n)}e^{{-{n}}}"


def transform_number(v):
    if np.abs(v) >= 0.001:
        return f"{v:.1f}"
    elif 0 < np.abs(v):
        return f"{transform_power_of_ten(v)}"
    else:
        return "0"


def mk_score_line(dict, method_name, f_name, print_std, best_score=False):
    v = transform_number(dict[method_name][f_name]["mean"])
    std = transform_number(dict[method_name][f_name]["std"])
    if best_score:
        if print_std:
            return f" \\makecell{{$\\mathbf{{{v}}}$ \\\\ $\\mathbf{{\\pm {std}}}$}} &"
        else:
            return f" \\makecell{{$\\mathbf{{{v}}}$}} &"
    else:
        if print_std:
            return f" \\makecell{{${v}$ \\\\ $\\pm {std}$}} &"
        else:
            return f" \\makecell{{${v}$}} &"


def get_best_score(dict, f_name, return_name=False):
    best_score = np.inf
    best_name = None
    for method_name in dict.keys():
        if dict[method_name][f_name]["mean"] < best_score:
            best_name = method_name
            best_score = dict[method_name][f_name]["mean"]
    if return_name:
        return best_score, best_name
    return best_score


def compute_ratio(sota_methods, proposed_methods, function_mins):
    ratios = {}
    for f_name in function_mins.keys():
        best_value = min(
            get_best_score(sota_methods, f_name),
            get_best_score(proposed_methods, f_name),
        )
        best_dist = np.abs(best_value - function_mins[f_name])

        ratios[f_name] = {}
        for s_name in sota_methods.keys():
            dist = np.abs(sota_methods[s_name][f_name]["mean"] - function_mins[f_name])
            ratios[f_name][s_name] = min(100, (dist + 1e-10) / (best_dist + 1e-10))

        for p_name in proposed_methods.keys():
            dist = np.abs(
                proposed_methods[p_name][f_name]["mean"] - function_mins[f_name]
            )
            ratios[f_name][p_name] = min(100, (dist + 1e-10) / (best_dist + 1e-10))
    return ratios


def mk_table(function_mins, sota_methods, proposed_methods, all_ranks, std):
    function_names = list(function_mins.keys())
    sota_names = sort_methods(list(sota_methods.keys()), all_ranks)
    proposed_names = sort_methods(list(proposed_methods.keys()), all_ranks)

    lines = [
        "functions"
        + " & "
        + " & ".join(sota_names)
        + " & "
        + " & ".join(proposed_names)
        + " \\\\"
    ]

    lines.append("\\midrule")

    for f_name in function_names:
        line = f"{f_name} &"
        sota_best = get_best_score(sota_methods, f_name, return_name=True)
        proposed_best = get_best_score(proposed_methods, f_name, return_name=True)
        bf_sota = sota_best[0] < proposed_best[0]

        for s_name in sota_names:
            if s_name == sota_best[1] and bf_sota:
                line += mk_score_line(
                    sota_methods, s_name, f_name, std, best_score=True
                )
            else:
                line += mk_score_line(sota_methods, s_name, f_name, std)
        for p_name in proposed_names:
            if p_name == proposed_best[1] and not bf_sota:
                line += mk_score_line(
                    proposed_methods, p_name, f_name, std, best_score=True
                )
            else:
                line += mk_score_line(proposed_methods, p_name, f_name, std)
        line = line[:-2]
        line += " \\\\"
        lines.append(line)

    lines.append("\\midrule")

    ratios_lines = "Comp. ratio"
    ratios = compute_ratio(sota_methods, proposed_methods, function_mins)
    sum_ratios = []
    for s_name in sota_names:
        sum_ratios.append(np.sum([ratios[f_name][s_name] for f_name in function_names]))
    for p_name in proposed_names:
        sum_ratios.append(np.sum([ratios[f_name][p_name] for f_name in function_names]))
    for ratio in sum_ratios:
        if ratio == np.min(sum_ratios):
            ratios_lines += (
                f" & $\\mathbf{{{transform_number(ratio / len(function_names))}}}$"
            )
        else:
            ratios_lines += f" & ${transform_number(ratio / len(function_names))}$"
    ratios_lines += " \\\\"
    lines.append(ratios_lines)
    lines.append("\\midrule")

    avg_line = "Average rank"
    mean_ranks = {}
    for name in all_ranks.keys():
        mean_ranks[name] = np.mean(all_ranks[name])
    for name in sota_names + proposed_names:
        rank = mean_ranks[name]
        if rank == np.min(list(mean_ranks.values())):
            avg_line += f" & $\\mathbf{{{rank:.2f}}}$"
        else:
            avg_line += f" & ${rank:.2f}$"
    avg_line += " \\\\"
    lines.append(avg_line)
    lines.append("\\midrule")

    sorted_method = sort_methods(sota_names + proposed_names, all_ranks)
    sorted_method.reverse()

    final_line = "Final rank"
    for name in sota_names + proposed_names:
        rank = sorted_method.index(name) + 1
        if rank == 1:
            final_line += " & $\\mathbf{1}$"
        else:
            final_line += f" & ${rank}$"
    final_line += " \\\\"
    lines.append(final_line)
    lines.append("\\bottomrule")

    print("\n\n")
    for l in lines:
        print_purple(l)
    print("\n\n")
