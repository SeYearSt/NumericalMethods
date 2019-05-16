import numpy as np
import matplotlib.pyplot as plt


def binary_search(f: 'function', a: float, b: float, eps: float = 1e-3) -> tuple:
	if b < a:
		raise ValueError("End of interval cannot be less than being")

	if f(b)*f(a) > 0:
		raise Exception("There is no root in passed interval.")

	iteration_count = 0

	middle = (a+b)/2
	while b - a >= eps:
		middle = (a + b)/2
		iteration_count += 1
		print("xl = {}, xx = {}, xp = {}".format(a, middle, b))
		if f(middle) * f(a) < 0:
			b = middle
		if f(middle) * f(b) < 0:
			a = middle
		if f(middle) == 0:
			break

	return iteration_count, middle


def simple_iter(x: float, eps: float) -> tuple:
	def fi(x: float) -> float:
		return np.power(x, 3)

	iteration_count = 0

	while True:
		x_new = fi(x)
		print("fi(x_{})=".format(iteration_count), x_new)
		variance = x_new - x
		x = x_new
		iteration_count += 1
		if abs(x - x_new) < eps:
			break

	return iteration_count, x_new, variance


def f(x: float) -> float:
	return np.sin(2*x) - x**2


def plot_f(f: 'function', a:float, b:float, n: int) -> float:
	X = np.linspace(a, b, n)
	Y = [f(x) for x in X]
	plt.grid()
	plt.plot(X, Y)
	plt.show()


def input_init_values() -> tuple:
	a, b = (float(x) for x in input("Input begin and end of interval: ").split())
	print("Your input is a={}, b={}".format(a, b))
	eps = float(input("Input epsilon: "))
	print("Your input is eps={}".format(eps))

	return a, b, eps


if __name__ == '__main__':
	debug = False

	# debug variables
	# a, b, eps, eps2 = -1, 0.5, 1e-3, 1e-5

	a, b, eps = input_init_values()

	if debug:
		plot_f(f, a, b, 100)

	iteration_binary, x0 = binary_search(f, a, b, eps)
	print("Binary search iteration count:", iteration_binary)
	print("Binary search result: ", x0)

	eps2 = float(input('Input epsilon for simple iteration: '))
	iteration_simple_iter, x1, variance = simple_iter(x0, eps2)
	print("Simple iteration iteration count:", iteration_simple_iter)
	print("Variance:", variance)
	print('Simple iteration result =', x1)
