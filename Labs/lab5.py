import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 1, 0.1)
Y = np.array([0., 0.22140, 0.49182, 0.82211, 1.22554, 1.71828, 2.32011, 3.05519, 3.95303, 5.04964])
# print(X.shape)
# print(Y.shape)


# X = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Y = [0., 0.22140, 0.49182, 0.82211, 1.22554, 1.71828, 2.32011, 3.05519, 3.95303, 5.04964]
#
# X_test = [0.405, 0.661, 0.822]


def divided_diff(x: np.ndarray, y: np.ndarray) -> float:
    if y.shape != y.shape:
        raise ValueError("Y and X should have the same shape, got X.shape = {}, Y.shape = {}".format(x.shape, y.shape))

    result = 0
    for j in range(y.shape[0]):
        indexes = np.arange(y.shape[0]) != j
        denominator = np.prod(x[j] - x[indexes])
        result += y[j]/denominator

    return result



if __name__ == '__main__':
    print(divided_diff(np.array([0., 4.]), np.array([0., 16.])))
    # divided_diff(X, Y)
    # plt.plot(X, Y, '*')
    # plt.show()
