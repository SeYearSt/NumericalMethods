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
        divided_diff_s = np.hstack([divided_diff_s, np.zeros(y.shape[0]-1-divided_diff_s.shape[0])])
        values.append(divided_diff_s)

    for row in zip(*values):
        tab.add_row(row)

    s = tab.draw()
    print(s)


if __name__ == '__main__':

    X = np.arange(0, 1, 0.1)
    Y = np.array([0., 0.22140, 0.49182, 0.82211, 1.22554, 1.71828, 2.32011, 3.05519, 3.95303, 5.04964])

    X_test = np.array([0.221, 0.428, 0.681, 0.3])

    for idx, node in enumerate(X):
        print("Value in table: {}, value of interpolation polygon: {}".format(Y[idx], L(node, X, Y)))
        if idx >= 3:
            break

    print()

    for x in X_test:
        print("Value of interpolation polynomal in {}: is {}".format(x, L(x, X, Y)))

    print_divided_diff_table(X, Y)

    debug = True

    if debug:
        fig = plt.figure()
        plt.plot(X, Y, "ob", markersize=5)
        X_interp = np.linspace(X[0], X[-1], 100)
        plt.plot(X_interp, [L(x, X, Y) for x in X_interp], 'oy', markersize=2)
        plt.legend(["Table point", "Interpolation"])
        plt.show()

