import numpy as np

def simple_iter(A: list, f: list, epsilon: float, rand: bool = True):
    iterations = 0
    
    if rand:
     x = np.reshape(np.random.rand(f.size, 1), (f.size, 1))
    else:
      x = np.zeros((f.size, 1))
    
    convergence = False
    
    while not convergence:
        x_new = x - np.matmul(A,x) + f
        
        convergence = np.linalg.norm(x_new - x) <= epsilon
    
        x = x_new
        iterations += 1
    
    return x, iterations, A@x - f
    
