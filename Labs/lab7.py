import numpy as np


def integral(f: 'function', a: float, b: float, N: int = 8) -> float:
    if a > b:
        raise ValueError("Begin of inteerval should be greater than end.")
    
    x = np.linspace(a, b, N)
    h = (b - a)/N
    result = np.sum(f(x)*h)
    return result

def integral_error(f: 'function', integral: 'function', a: float, b: float, N: int = 100) -> float:
    integral_h1 = integral(f, a, b, N)
    integral_h2 = integral(f, a, b, N//2)
    error = (integral_h1 - integral_h2)/3
    return error

def integral_precision(f: 'function', integral: 'function', a: float, b: float, eps: float) -> float:
    p = 2
    N = 8
    while True:
        h = (b-a)/N
        x = np.linspace(a, b, N)
        res1 = integral(f, a, b, N)
        res2 = integral(f, a, b, 2*N)
        print('N={}, h={}, I={}'.format(N, h, res1))

        if np.abs(res1 - res2) <= (2**p - 1)*eps:
            break
        N *= 2

    res = res2 + (res2 - res1)/(2**p - 1)
    return res


def f(x: float) -> float:
	return  np.power(np.tan(x/2 + np.pi/4), 3)

if __name__ == '__main__':
    a, b = 0, np.pi/4
    eps = 1e-3
    print('Defined integral of {}, a={}, b={} is: {}'.format('tan(x/2 + pi/4)**3', a, b, integral_precision(f, integral, a, b, eps)))
    print('Error', integral_error(f, integral, a, b))
