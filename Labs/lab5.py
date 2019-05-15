import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 1, 0.1)
Y = np.array([0., 0.22140, 0.49182, 0.82211, 1.22554, 1.71828, 2.32011, 3.05519, 3.95303, 5.04964])

X_test = np.array([0.405, 0.661, 0.822])


def divided_diff(x: np.ndarray, y: np.ndarray) -> float:
    if y.shape != y.shape:
        raise ValueError("Y and X should have the same shape, got X.shape = {}, Y.shape = {}".format(x.shape, y.shape))

    result = 0
    for j in range(y.shape[0]):
        indexes = np.arange(y.shape[0]) != j
        denominator = np.prod(x[j] - x[indexes])
        result += y[j]/denominator

    return result


def L(x: float) -> float:
    result = Y[0]
    for i in range(1, Y.shape[0]):
        indexes = np.arange(i)
        indexes_diff = np.append(indexes, indexes[-1]+1)
        result += np.prod(x - X[indexes])*divided_diff(X[indexes_diff], Y[indexes_diff])

    return result


if __name__ == '__main__':

    for idx, node in enumerate(X):
        print("Value in table: {}, value of interpolation polygon: {}".format(Y[idx], L(node)))
        if idx >= 3:
            break

    print()

    for x in X_test:
        print("Value of interpolation polynomal in {}: is {}".format(x, L(x)))

    debug = False

    if debug:
        fig = plt.figure()
        plt.plot(X, Y, "ob", markersize=5)
        X_interp = np.linspace(X[0], X[-1], 100)
        plt.plot(X_interp, [L(x) for x in X_interp], 'oy', markersize=2)
        plt.legend(["Fact", "Interpolation"])
        plt.show()

