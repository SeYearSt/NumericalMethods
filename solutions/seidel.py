from math import sqrt

def seidel(A, b, eps):
    n = len(A)
    x = [.0 for i in range(n)]

    converge = False
    while not converge:
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = x_new

    return x

if __name__ == '__main__':
     A = [[0.45, 0.03, -0.01, 0.02, -0.111], [0.02, 0.375, -0.01, -0.01, 0], [0, 0.07, 0.44, 0, 0.113], [-0.03, 0.015, -0.02, 0.41, -0.084], [0.02, 0.01, 0, 0, 0.29]]
b = [-0.023, -0.69, 0.199, -1.952, 0]
print(seidel(A, b, 0.1))
