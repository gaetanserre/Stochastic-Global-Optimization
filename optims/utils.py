def print_color(str, color):
    print(f"\033[{color}m" + str + "\033[0m")


print_pink = lambda str: print_color(str, 95)
print_blue = lambda str: print_color(str, 94)
print_green = lambda str: print_color(str, 92)
print_purple = lambda str: print_color(str, 35)
