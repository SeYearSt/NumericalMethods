import matplotlib.pyplot as plt
import numpy as np
import texttable as tt


def divided_diff(x: np.ndarray, y: np.ndarray) -> tuple:
    n = y.shape[0]

    result = [[] for i in range(n)]
    result[0] = y

    for k in range(n-1):
        for i in range(1, len(result[k])):
            divided_diff = (result[k][i]-result[k][i-1])/(x[i+k] - x[i-1])
            result[k+1].append(divided_diff)

    coefs = [diff[0] for diff in result]

    return coefs, result


def L(x: float, x_i: list, f_diff: list) -> float:
    x_i = np.insert(x_i, 0, x-1)
    res = f_diff[-1]
    res *= (x-x_i[-1])
    for i in range(len(f_diff)-1, -1, -1):
        res += f_diff[i]
        res *= x-x_i[i]

    return res


def print_divided_diff_table(x: np.ndarray, y: np.ndarray, f_coefs: list) -> None:
    f_coefs_copy = [[val for val in f_coef] for f_coef in f_coefs]

    tab = tt.Texttable()

    # ----------- set headers
    headings = ['X', 'Y']
    for i in range(1, y.shape[0]):
        head = 'f_x_i...f_x_i+{}'.format(i)
        headings.append(head)

    tab.header(headings)

    values = [x]

    # --------- cal divided differences

    for i in range(len(f_coefs_copy)):
        f_coefs_copy[i].extend([""]*(y.shape[0]-len(f_coefs_copy[i])))
        values.append(f_coefs_copy[i])

    for row in zip(*values):
        tab.add_row(row)

    s = tab.draw()
    print(s)


def f(x: float) -> float:
    return np.cosh(x/2)/10


def print_polynomial(X, f_coefs) -> None:
    template = '(x-{:.3f})'
    polinomial = 'Ln(x) = {:.3f}+'
    args = [f_coefs[0]]
    for i in range(1, X.shape[0]):
        polinomial += template*i + '{:.3f}+'
        args.extend(X[:i].tolist())
        args.append(f_coefs[i])

    polinomial = polinomial[:-1]

    print(polinomial.format(*args))


def print_difference(X_test, X, Y) -> None:
    L_x = np.array([L(x, X, Y) for x in X_test])
    difference = f(X_test) - L_x

    tab = tt.Texttable()
    headings = ['x', 'f(x)', 'L(x)', 'R(x)']
    tab.header(headings)
    values = [X_test, f(X_test), L_x, difference]
    for row in zip(*values):
        tab.add_row(row)
    s = tab.draw()
    print(s)


if __name__ == '__main__':

    a, b = 0, 3
    n = 5
    X = np.linspace(a, b, n)
    Y = f(X)

    print(X)
    print(Y)

    f_coefs, divided_diffs = divided_diff(X, Y)

    # print(L(0, X, f_coefs))

    X_test = np.array([1.2, 2., 2.8, 0.5, 0.75, 3.4, 10])

    print("Netwon's polynomial")
    print_polynomial(X, f_coefs)
    print("Table of divided differences")
    print_divided_diff_table(X, Y, divided_diffs)
    print('Difference between function and interpolation polynomial')
    print_difference(X_test, X, Y)

    debug = True

    if debug:
        fig = plt.figure()
        plt.plot(X, Y, "ob", markersize=5)
        X_interp = np.linspace(X[0], X[-1], 100)
        plt.plot(X_interp, [L(x, X, f_coefs) for x in X_interp], 'oy', markersize=2)
        plt.legend(["Table point", "Interpolation"])
        plt.show()
