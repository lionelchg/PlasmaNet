import numpy as np

def fwd_row(L, b):
    n = len(b)
    x = np.zeros_like(b)
    x[0] = b[0] / L[0, 0]
    for i in range(1, n):
        x[i] = (b[i] - np.sum(L[i, :i] * x[:i])) / L[i, i]
    return x

def fwd_col(L, b):
    n = len(b)
    for i in range(n - 1):
        b[i] /= L[i, i]
        b[i+1:] -= b[i] * L[i+1:, i]
    b[-1] /= L[-1, -1]
    return b

def bwd_row(U, b):
    n = len(b)
    x = np.zeros_like(b)
    x[-1] = b[-1] / U[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - np.sum(U[i, i+1:] * x[i+1:])) / U[i, i]
    return x

def bwd_col(U, b):
    n = len(b)
    for i in range(n - 1, 0, -1):
        b[i] /= U[i, i]
        b[:i-1] -= b[i] * U[:i-1, i]
    b[0] /= U[0, 0]
    return b

if __name__ == '__main__':
    n = 10
    U, L = np.zeros((n, n)), np.zeros((n, n))
    bu, bl = np.zeros(n), np.zeros(n)
    for i in range(n):
        L[i, :i+1] = 1
        U[i, i:] = 1
        bl[i] = i + 1
        bu[i] = n - i
        
    print(fwd_row(L, bl))
    print(fwd_col(L, bl))
    print(bwd_row(U, bu))
    print(bwd_row(U, bu))
