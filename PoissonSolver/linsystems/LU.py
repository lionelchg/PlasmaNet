import numpy as np
import copy
from utils import print_mat

def LU_kji(A):
    n = len(A[:, 0])
    for k in range(n - 1):
        A[k + 1:, k] /= A[k, k]
        A[k + 1:, k + 1:] -= A[k + 1:, k, np.newaxis].dot(A[np.newaxis, k, k + 1:])
    return A

def fwd_row(L, b):
    n = len(b)
    x = np.zeros_like(b)
    # L[i, i] is implicitely equal to 1 in the routines
    x[0] = b[0]
    for i in range(1, n):
        x[i] = (b[i] - np.sum(L[i, :i] * x[:i]))
    return x

def bwd_row(U, b):
    n = len(b)
    x = np.zeros_like(b)
    x[-1] = b[-1] / U[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - np.sum(U[i, i+1:] * x[i+1:])) / U[i, i]
    return x

def solve_system(LU, b):
    return bwd_row(LU, fwd_row(LU, b))

if __name__ == '__main__':
    n = 10
    L, U = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        L[i:, i] = np.linspace(1, n - i, n - i)
        U[i, i:] = 1
    print_mat(L, 'L')
    print_mat(U, 'U')
    A = L.dot(U)
    print_mat(A, 'LU')
    LU_dec = copy.deepcopy(A)
    print_mat(LU_kji(LU_dec), 'LU decomp')

    x = np.linspace(1, n, n)
    f = lambda x: x**2
    y = f(x)
    
    b = A.dot(y)

    print('y_exact', y)

    y_solve = solve_system(LU_dec, b)

    print('y_solve', y_solve)