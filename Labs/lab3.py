import numpy as np


def f(x: float) -> float:
	return np.sin(2*x)-x**2

def binary_search(f: 'function', a: float, b: float, eps: float = 1e-3) -> float:
	while b - a >= eps:
		middle = (a + b)/2
		if f(middle) * f(a) < 0:
			b = middle
		if f(middle) * f(b) < 0:
			a = middle
		if f(middle) == 0:
			return middle
	return middle


if __name__ == '__main__':
	a, b = -0.5, 1
	print(binary_search(f, a, b))
