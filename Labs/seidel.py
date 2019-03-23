import numpy as np
import helpers

def seidel(A: list, f:list, eps: float, rand: bool = True) -> tuple:
    n = len(A)    

    P, L, U = helpers.lu_decompose(A)
    #print('P={}\nL={}\nU={}'.format(P, L, U))

    if rand:
     x = np.random.rand(f.size, 1)
    else:
      x = np.zeros((f.size, 1))

    iterations = 0
    convergence = False	
  
    while not convergence:
        x_new = x.copy()

        s1 = np.matmul(L, x_new)
        s2 = np.matmul(U, x)
        x_new = (f - s1 - s2)/P
         
        convergence = np.linalg.norm(x_new - x) <= eps

        x = x_new
        iterations += 1

    difference = np.matmul(A,x) - f

    return x, iterations, difference

