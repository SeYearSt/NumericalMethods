import matplotlib.pyplot as plt
import numpy as np
import texttable as tt


def divided_diff(x: np.ndarray, y: np.ndarray) -> float:
    if y.shape != y.shape:
        raise ValueError("Y and X should have the same shape, got X.shape = {}, Y.shape = {}".format(x.shape, y.shape))

    result = 0
    for j in range(y.shape[0]):
        indexes = np.arange(y.shape[0]) != j
        denominator = np.prod(x[j] - x[indexes])
        result += y[j]/denominator

    return result


def L(x: float, x_i: np.ndarray, y_i: np.ndarray) -> float:
    result = y_i[0]
    for i in range(1, y_i.shape[0]):
        indexes = np.arange(i)
        indexes_diff = np.append(indexes, indexes[-1]+1)
        result += np.prod(x - x_i[indexes])*divided_diff(x_i[indexes_diff], y_i[indexes_diff])

    return result


def print_divided_diff_table(x: np.ndarray, y: np.ndarray) -> None:
    tab = tt.Texttable()

    # ----------- set headers
    headings = ['X', 'Y']
    for i in range(1, y.shape[0]):
        head = 'f_x_i...f_x_i+{}'.format(i)
        headings.append(head)

    tab.header(headings)

    values = [x, y]

    # --------- cal divided differences

    for i in range(1, y.shape[0]):
        divided_diff_s = np.array([divided_diff(x[j: j+i+1], y[j: j+i+1]) for j in range(y.shape[0] - i)])
        divided_diff_s = np.hstack([divided_diff_s, [""]*(y.shape[0]-1-divided_diff_s.shape[0])])
        values.append(divided_diff_s)

    for row in zip(*values):
        tab.add_row(row)

    s = tab.draw()
    print(s)


def f(x: float) -> float:
    return np.cosh(x/2)/10
	

def print_polynomial(X, Y) -> None:
    template = '(x-{:.3f})'
    polinomial = 'Ln(x) = {:.3f}+'
    args = [L(X[0], X, Y)]
    for i in range(1, X.shape[0]):
        polinomial += template*i + '{:.3f}+'
        args.extend(X[:i].tolist())
        args.append(divided_diff(X[:i+1], Y[:i+1]))

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

    X_test = np.array([1.2, 2., 2.8, 0.5, 0.75, 3.4, 10])

    print("Netwon's polynomial")
    print_polynomial(X, Y)
    print("Table of divided differences")
    print_divided_diff_table(X, Y)
    print('Difference between function and interpolation polynomial')
    print_difference(X_test, X, Y)

    debug = False

    if debug:
        fig = plt.figure()
        plt.plot(X, Y, "ob", markersize=5)
        X_interp = np.linspace(X[0], X[-1], 100)
        plt.plot(X_interp, [L(x, X, Y) for x in X_interp], 'oy', markersize=2)
        plt.legend(["Table point", "Interpolation"])
        plt.show()
