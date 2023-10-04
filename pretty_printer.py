#
# Created in 2023 by Gaëtan Serré
#

from prettytable import PrettyTable


def pprint_results(dict_optim_res):
    table = [["Algorithms", "Average value", "Average eval", "Average time (s)"]]
    for opt_name, values in dict_optim_res.items():
        table.append([opt_name] + values)
    tab = PrettyTable(table[0])
    tab.add_rows(table[1:])
    print(tab)
