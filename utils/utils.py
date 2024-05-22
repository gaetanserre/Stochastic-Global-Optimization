#
# Created in 2024 by Gaëtan Serré
#

import numpy as np


def print_color(str, color):
    print(f"\033[{color}m" + str + "\033[0m")


print_pink = lambda str: print_color(str, 95)
print_blue = lambda str: print_color(str, 94)
print_green = lambda str: print_color(str, 92)
print_purple = lambda str: print_color(str, 35)


def time_it(function, args={}):
    import time

    start = time.time()
    ret = function(**args)
    end = time.time()
    return ret, end - start


def merge_ranks(rank_dict, new_ranks):
    if rank_dict == {}:
        return new_ranks
    else:
        for k, v in rank_dict.items():
            rank_dict[k] = v + new_ranks[k]
        return rank_dict


def add_dict_entry(dict, key, value):
    if key not in dict:
        dict[key] = value
    else:
        dict[key].update(value)


def transform_dict_to_opt_dict(return_dict):
    value_dict = {}
    time_dict = {}
    eval_dict = {}

    def add_dict(d, k, v):
        if k not in d:
            d[k] = [v]
        else:
            d[k].append(v)

    for l in return_dict.values():
        for k, v in l.items():
            add_dict(value_dict, k, v[0])
            add_dict(eval_dict, k, v[1])
            add_dict(time_dict, k, v[2])
    opt_dict = {}
    for k in value_dict.keys():
        opt_dict[k] = [
            np.mean(value_dict[k]),
            f"{np.mean(eval_dict[k]):.2f}",
            f"{np.mean(time_dict[k]):.4f}",
        ]
    return opt_dict


def create_bounds(min, max, dim):
    bounds = [(min, max) for _ in range(dim)]
    return np.array(bounds)
