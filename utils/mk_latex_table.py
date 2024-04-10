#
# Created in 2024 by Gaëtan Serré
#

import numpy as np

# dict : Method name -> Function name -> score


def sort_methods(methods, all_ranks):
    key = lambda name: -np.mean(all_ranks[name])
    methods.sort(key=key)
    return methods


def transform_power_of_ten(v):
    n = 0
    while v * 10**n <= 1:
        n += 1
    return f"${v * 10**n:.2f} \\times 10^{{-{n}}}$"


def transform_number(v):
    if np.abs(v) >= 0.001:
        return f"${v:.3f}$"
    elif 0 < np.abs(v):
        return f"{transform_power_of_ten(v)}"
    else:
        return "$0$"


def mk_score_line(dict, method_name, f_name):
    v = dict[method_name].get(f_name)
    return f" {transform_number(v)} &"


def get_best_score(dict, f_name):
    best_score = np.inf
    for method_name in dict.keys():
        best_score = min(best_score, dict[method_name][f_name])
    return best_score


def compute_ratio(sota_methods, proposed_methods, function_names):
    ratios = {}
    for f_name in function_names:
        best_value = get_best_score(sota_methods, f_name)
        best_value = min(best_value, get_best_score(proposed_methods, f_name)) + 1e-50
        ratios[f_name] = {}
        for s_name in sota_methods.keys():
            ratios[f_name][s_name] = best_value / sota_methods[s_name][f_name] + 1e-50
        for p_name in proposed_methods.keys():
            ratios[f_name][p_name] = (
                best_value / proposed_methods[p_name][f_name] + 1e-50
            )
    return ratios


def mk_table(function_names, sota_methods, proposed_methods, all_ranks):
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
        for s_name in sota_names:
            print(s_name, f_name, sota_methods[s_name][f_name])
            line += mk_score_line(sota_methods, s_name, f_name)
        for p_name in proposed_names:
            line += mk_score_line(proposed_methods, p_name, f_name)
        line = line[:-2]
        line += " \\\\"
        lines.append(line)

    lines.append("\\midrule")

    ratios_lines = "$\\frac{1}{|F|} \sum_{f \in F} \\text{sol}_f / \\text{best}_f$"
    ratios = compute_ratio(sota_methods, proposed_methods, function_names)
    sum_ratios = []
    for s_name in sota_names:
        sum_ratios.append(np.sum([ratios[f_name][s_name] for f_name in function_names]))
    for p_name in proposed_names:
        sum_ratios.append(np.sum([ratios[f_name][p_name] for f_name in function_names]))

    for ratio in sum_ratios:
        ratios_lines += f" & {transform_number(ratio / len(function_names))}"
    ratios_lines += " \\\\"
    lines.append(ratios_lines)
    lines.append("\\midrule")

    avg_line = "Average rank"
    for s_name in sota_names:
        avg_line += f" & {np.mean(all_ranks[s_name]):.2f}"
    for p_name in proposed_names:
        avg_line += f" & {np.mean(all_ranks[p_name]):.2f}"
    avg_line += " \\\\"
    lines.append(avg_line)
    lines.append("\\midrule")

    sorted_method = sort_methods(sota_names + proposed_names, all_ranks)
    sorted_method.reverse()

    final_line = "Final rank"
    for s_name in sota_names:
        final_line += f" & {sorted_method.index(s_name) + 1}"
    for p_name in proposed_names:
        final_line += f" & {sorted_method.index(p_name) + 1}"
    final_line += " \\\\"
    lines.append(final_line)
    lines.append("\\bottomrule")

    for l in lines:
        print(l)
