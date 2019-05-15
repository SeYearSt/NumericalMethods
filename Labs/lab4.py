import numpy as np

from Labs.lab3 import binary_search
from Labs.lab3 import input_init_values
from Labs.lab3 import plot_f
from Labs.lab3 import f


def f_d(x: float) -> float:
    return 2*np.cos(2*x) - 2*x


def newton(f: 'function', f_d: 'function', x: float, eps: float) -> tuple:
    iteration_count = 0

    while True:
        x_new = x - f(x)/f_d(x)
        iteration_count += 1
        print("x_new_{}=".format(iteration_count), x_new)
        if abs(x_new - x) <= eps:
            break
        x = x_new

    return iteration_count, x_new, x_new - x


if __name__ == '__main__':
    debug = True

    # debug variables
    # a, b, eps, eps2 = -1, 0.5, 1e-3, 1e-5

    a, b, eps = input_init_values()

    if debug:
        plot_f(f, a, b, 100)

    iteration_binary, x0 = binary_search(f, a, b, eps)
    print("Binary search iteration count:", iteration_binary)
    print("Binary search result: ", x0)

    eps2 = float(input('Input epsilon for simple iteration: '))
    iterations_newton, x1, variance = newton(f, f_d, x0, eps2)
    print("Newton's  number of iterations:", iterations_newton)
    print("Variance:", variance)
    print('Newton result =', x1)
