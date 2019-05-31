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
        head = 'f(x_i;...;x_i+{})'.format(i)
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

  polinomial = 'Ln(x) = {:.3f}+'

  template = '(x-{:.3f})({:.3f}*'
  template_end = '(x-{:.3f}){:.3f}'
  args = []

  for i in range(len(f_coefs)-2):
    polinomial += template
  
  polinomial += template_end
  polinomial += ")"*(len(f_coefs)-2)

  for i in range(0, len(f_coefs)):
    args.append(f_coefs[i])
    args.append(X[i])

  print(polinomial.format(*args))


def print_difference(X_test, analytic_difference, X, Y, x_diff) -> None:
    L_x = np.array([L(x, X, x_diff) for x in X_test])
    f_res = f(X_test)
    difference = abs(f_res - L_x)

    tab = tt.Texttable()
    headings = ['x', 'f(x)', 'L(x)', 'R(x)', 'teoretical R(x)']
    tab.header(headings)
    values = [X_test, f_res, L_x, difference, analytic_difference]
    for row in zip(*values):
        tab.add_row(row)
    s = tab.draw()
    print(s)


if __name__ == '__main__':

    a, b = 0, 3
    n = 4
    X = np.linspace(a, b, n+1)
    Y = f(X)

    f_coefs, divided_diffs = divided_diff(X, Y)

    X_test = np.array([0.3, 0.5, 1.75, 0.75])
    analytic_difference = [0.0047, 0.003,  0.0015, 0.]

    print("Netwon's polynomial")
    print_polynomial(X, f_coefs)
    print("Table of divided differences")
    print_divided_diff_table(X, Y, divided_diffs)
    print('Difference between function and interpolation polynomial')
    print_difference(X_test, analytic_difference, X, Y, f_coefs)

    debug = False
    if debug:
        fig = plt.figure()
        plt.plot(X, Y, "ob", markersize=5)
        X_interp = np.linspace(X[0], X[-1], 100)
        plt.plot(X_interp, [L(x, X, f_coefs) for x in X_interp], 'oy', markersize=2)
        plt.legend(["Table point", "Interpolation"])
        plt.show()