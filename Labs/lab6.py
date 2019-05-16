from sklearn.linear_model import LinearRegression

from Labs.lab5 import *


def square_divergence(L: "linear model", x: np.ndarray, y: np.ndarray) -> np.ndarray:
    interp = L.predict(x)
    result = np.power(interp - y, 2)
    return result


def print_divergence_table(L: 'function', x: np.ndarray, y:np.ndarray) -> None:
    tab = tt.Texttable()

    # ----------- set headers
    headings = ['X', 'Y', 'Diff']
    tab.header(headings)

    diff = square_divergence(L, x, y)

    for row in zip(x, y, diff):
        tab.add_row(row)

    s = tab.draw()
    print(s)


if __name__ == '__main__':
    X = np.arange(0, 1, 0.1)
    Y = np.array([0., 0.22140, 0.49182, 0.82211, 1.22554, 1.71828, 2.32011, 3.05519, 3.95303, 5.04964])

    X_train = X.reshape(-1, 1)
    X_train = X_train

    Y_train = Y.reshape(-1, 1)
    Y_train = Y_train

    lm = LinearRegression()
    lm.fit(X_train, Y_train)

    X_test = np.linspace(X[0], X[-1], 100).reshape(-1, 1)

    print_divergence_table(lm, X_train, Y_train)
    print(np.sum(square_divergence(lm, X_train, Y_train)))


    # TODO: add title for axes

    debug = True
    if debug:
        fig = plt.figure()
        locs, labels = plt.xticks()
        plt.grid()
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.plot(X, Y, "ob", markersize=5)
        plt.plot(X_test, lm.predict(X_test), markersize=2)
        plt.legend(["Table point", "Interpolation"])
        plt.show()