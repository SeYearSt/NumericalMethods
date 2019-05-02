import numpy as np
import matplotlib.pyplot as plt

def binary_search(f: 'function', a: float, b: float, eps: float = 1e-3) -> float:
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
			return middle
	print('Binary search number of iterations is:',iteration_count)
	return middle


def simple_iter(x:float, eps: float) -> float:
	iteration_count = 0
	while True:
		x_new = fi(x)
		#print(x)
		#print('len=', abs(x_new - x))

		if abs(x_new - x) >= eps:
			break
		iteration_count += 1
		#print(x)
		x = x_new
	print('Simple iteration method number of iteration', iteration_count)
	return x


def f(x: float) -> float:
	return np.sin(2*x) - x**2

def ro(x: float) -> float:
	return 0.01

def fi(x: float) -> float:
	return x + ro(x)*f(x)


#def f_der(x: float) -> float:
#	return 2*np.cos(2*x) - 2*x 


#def f_der_second(x: float) -> float:
#	return -4*np.sin(2*x) - 2


def newton(f: 'function', f_der: 'function', x: float, eps: float = 1e-5) -> float:
	while True:
		x_next = x - f(x)/f_der(x)
		if abs(x_next - x) <= eps:
			return x_next
		x = x_next

def proof(f, f_der, f_der_second, a: float, b: float, n: int):
	def fi(x):
		return (f(x)*f_der_second(x))/(f_der(x)**2)

	x = np.linspace(a, b, n)
	print('X=',x)
	y = [fi(x_i) for x_i in x]
	print('Y=',y)
	plt.plot(x, y)
	plt.show()

def proof_simple_iter(a, b, n = 100):
	x = np.linspace(a, b, n)
	y = np.array([f_der(x_i) for x_i in x])
	y *= ro(x)
	y += 1
	plt.plot(x, y)
	plt.show()

def plot_f(f: 'function', a:float, b:float, n: int) -> float:
	x = np.linspace(a, b, n)
	y = np.sin(2*x) - x*x
	plt.grid()
	plt.plot(x, y)
	plt.show()


if __name__ == '__main__':
	#a, b = (float(x) for x in input("Input begin and end of interval: ").split())
	#print("Your input is a={}, b={}".format(a, b))
	#eps = float(input("Input epsilon: "))
	#print("Your input is eps={}".format(eps))
	a, b, eps, eps2 = 0.1, 0.8, 1e-3, 1e-5
	plot_f(f, a, b, 100)
	x0 = binary_search(f, a, b, eps)
	print("Binary search result: ", x0)
	#eps = float(input('Input epsilon for simple iteration: '))
	#proof_simple_iter(a, b)
	x1 = simple_iter(x0, eps2)
	print('Simple iteration result =', x1)
	#print(newton(f, f_der, x0))
	#proof(f, f_der, f_der_second, a, b, 5)
