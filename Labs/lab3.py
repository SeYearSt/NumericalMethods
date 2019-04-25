import numpy as np
import matplotlib.pyplot as plt

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


def simple_iter(f : 'function', x:float, eps: float = 1e-5) -> float:
	def fi(x: float) -> float:
		return x + f(x)
	while abs( x - fi(x)) >= eps:
		x = fi(x)
		print(x)
		print('len=', abs( x - fi(x)))
	return x


def f(x: float) -> float:
	return np.sin(2*x) - x**2


def f_der(x: float) -> float:
	return 2*np.cos(2*x) - 2*x 


def f_der_second(x: float) -> float:
	return -4*np.sin(2*x) - 2



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


def plot_f(f: 'function', a:float, b:float, n: int) -> float:
	x = np.linspace(a, b, n)
	y = np.sin(2*x) - x*x
	plt.grid()
	plt.plot(x, y)
	plt.show()


if __name__ == '__main__':
	a, b = -0.01, 0.02
	x0 = binary_search(f, a, b)
	print(newton(f, f_der, x0))
	#plot_f(f, a, b, 100)
	proof(f, f_der, f_der_second, a, b, 5)
