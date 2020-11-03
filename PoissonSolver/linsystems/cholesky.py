import numpy as np
import copy
from utils import print_mat

def cholesky(A):
    H = np.zeros_like(A)
    H[0, 0] = np.sqrt(A[0, 0])
    for j in range(1, n):
        for i in range(j):
            H[i, j] = (A[i, j] - np.sum(H[:i, j] * H[:i, i])) / H[i, i]
        H[j, j] = (A[j, j] - np.sum(H[:j, j]**2))**0.5
    return H

if __name__ == '__main__':
    n = 10
    H = np.zeros((n, n))
    for i in range(n):
        # H[i, i:] = np.linspace(1, n - i, n - i)
        H[:i+1, i] = np.linspace(1, i + 1, i + 1) 
    print_mat(H, 'H')
    Ht = np.transpose(H)
    print_mat(Ht, 'Ht')
    A = Ht.dot(H)
    print_mat(A, 'A')
    H_chol = cholesky(A)
    print_mat(H_chol, 'Hchol')