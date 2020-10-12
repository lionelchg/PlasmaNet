from scipy.sparse.linalg import spsolve, isolve, cg
from scipy import sparse
import numpy as np

if __name__ == '__main__':
    n_points = 101

    A = sparse.csc_matrix(
        sparse.dia_matrix((np.linspace(1, n_points, n_points), [0]), shape=(n_points, n_points)))

    print(A * 0.5)
    A[1, 0] = 10
    A[:, 0] = x.reshape(101, 1)
    print(A[:, 0])


    b = np.ones(n_points)

    x_direct = spsolve(A, b)

    x_iter = cg(A, b, tol=1e-8)
