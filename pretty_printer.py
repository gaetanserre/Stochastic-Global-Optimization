#
# Created in 2023 by Gaëtan Serré
#

from prettytable import PrettyTable


def pprint_results(dict_optim_res):
    first_row = ["Algorithms", "Average value", "Average eval", "Average time (s)"]
    tab = PrettyTable(first_row)

    rows = []
    for opt_name, values in dict_optim_res.items():
        rows.append([opt_name] + values)

    key = lambda row: row[1]
    rows.sort(key=key)

    for i in range(len(rows)):
        rows[i][0] = f"{i+1}. " + rows[i][0]

    tab.add_rows(rows)
    print(tab)
