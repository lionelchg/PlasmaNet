########################################################################################################################
#                                                                                                                      #
#                                        Routines concerning the linear system                                         #
#                                                                                                                      #
#                                      Lionel Cheng, Ekhi Ajuria CERFACS, 09.10.2020                                   #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from scipy import sparse
import scipy.constants as co



def laplace_square_matrix_reduced(n_points_orig):

    n_points = n_points_orig 
    diags = np.zeros((5, n_points * n_points))

    # Filling the diagonal
    diags[0, :] = - 4 * np.ones(n_points ** 2)
    diags[1, :] = np.ones(n_points** 2)
    diags[2, :] = np.ones(n_points** 2)
    diags[3, :] = np.ones(n_points** 2)
    diags[4, :] = np.ones(n_points** 2)
    # Creating the matrix
    # Definition sparse
    # dia_matrix((data, offsets), shape=(M, N))
    # where the data[k,:] stores the diagonal entries for diagonal offsets[k] (See example below)

    # Example
    # from scipy.sparse import dia_matrix
    # n = 10
    # ex = np.ones(n)
    # data = np.array([ex, 2 * ex, ex])
    # offsets = np.array([-1, 0, 1])
    # dia_matrix((data, offsets), shape=(n, n)).toarray()
    # array([[2., 1., 0., ..., 0., 0., 0.],
    #       [1., 2., 1., ..., 0., 0., 0.],
    #       [0., 1., 2., ..., 0., 0., 0.],
    #       ...,
    #       [0., 0., 0., ..., 2., 1., 0.],
    #       [0., 0., 0., ..., 1., 2., 1.],
    #       [0., 0., 0., ..., 0., 1., 2.]])
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, n_points, -n_points]), shape=(n_points** 2, n_points ** 2)))


def dirichlet_bc_reduced(rhs_orig, n_points_orig, up, down, left, right):

    n_points = n_points_orig - 2 
    rhs = rhs_orig[1:-1,1:-1]

    # filling of the four boundaries, inversion of up and down ?
    rhs[:n_points] -= up[1:-1]
    rhs[n_points * (n_points - 1):] -= down[1:-1]
    rhs[:n_points * (n_points - 1) + 1:n_points] -= left[1:-1]
    rhs[n_points - 1::n_points] -= right[1:-1]
    # mean approximation not necessary as the corner points shouldn't be used ...
    return rhs


