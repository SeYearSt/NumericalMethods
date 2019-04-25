
import numpy as np


def run(method):
    
    A = np.array([[0.405, 0.05, 0.04, 0, 0.09],[-0.061, 0.53, 0.073, 0.11, -0.06],[0.07, -0.036, 0.38, 0.03, 0.02],[-0.05, 0, 0.066, 0.58, 0.23],[0, 0.081, -0.05, 0, 0.41]])
    f = np.reshape(np.array([1.77, -0.53, -0.626, -2.772, 1.001]), (A.shape[1], 1))
    
    print('A=',A)
    print('f=', f)
    
    epsilon = float(input('Input epsilon: '))

    print('Pecision is:', epsilon)

    X, iterations, difference = method(A, f, epsilon)
    
    print('X')
    for idx, x in enumerate(X):
      print('X{}= '.format(idx+1), end='')
      print('{0:.10f}'.format(*x))

    print('Iterations=', iterations)
    
    print('Difference:')
    for idx, x in enumerate(difference):
      print('R[X{}]= '.format(idx+1), end='')
      print('{0:.10f}'.format(*x))


def lu_decompose(A: list):
    rows, columns = len(A), len(A[0])
    L = np.zeros((rows, columns))
    U = np.zeros((rows, columns))
    P = np.empty((rows, 1))

    for i in range(rows):
        for j in range(columns):
            if j < i:
                L[i][j] = A[i][j]
            elif j > i:
                U[i][j] = A[i][j]
            else:
                P[i][0] = A[i][j]

    return P, L, U


