#
# Created in 2023 by Gaëtan Serré
#

from prettytable import PrettyTable
import numpy as np
from copy import deepcopy


def pprint_results_get_rank(dict_optim_res):
    first_row = ["Algorithms", "Average value", "Average eval", "Average time (s)"]
    tab = PrettyTable(first_row)

    rows = []
    for opt_name, values in dict_optim_res.items():
        rows.append([opt_name] + values)

    key = lambda row: row[1]
    rows.sort(key=key)

    rows_c = deepcopy(rows)
    for i in range(len(rows)):
        rows_c[i][0] = f"{i+1}. " + rows[i][0]

    tab.add_rows(rows_c)
    print(tab)

    rank_dict = {}
    for i in range(len(rows)):
        rank_dict[rows[i][0]] = [i + 1]

    return rank_dict


def pprint_rank(all_ranks):
    first_row = ["Algorithms", "Final rank", "Average rank"]
    tab = PrettyTable(first_row)

    rows = []
    for k in all_ranks.keys():
        rows.append(k)

    key = lambda row: np.mean(all_ranks[row])
    rows.sort(key=key)

    for i in range(len(rows)):
        rows[i] = [rows[i], i + 1, f"{np.mean(all_ranks[rows[i]]):.3f}"]

    tab.add_rows(rows)
    print(tab)
