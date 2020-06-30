########################################################################################################################
#                                                                                                                      #
#                                        Routines concerning the linear system                                         #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from scipy import sparse
import scipy.constants as co

def laplace_square_matrix(n_points):
    diags = np.zeros((5, n_points * n_points))

    # Filling the diagonal
    diags[0, :] = - 4 * np.ones(n_points ** 2)
    diags[1, :] = np.ones(n_points ** 2)
    diags[2, :] = np.ones(n_points ** 2)
    diags[3, :] = np.ones(n_points ** 2)
    diags[4, :] = np.ones(n_points ** 2)

    # Correcting the diagonal to take into account dirichlet boundary conditions, first diagonal to one and correct 
    # all the other ones
    for i in range(n_points ** 2):
        if i < n_points or i >= (n_points - 1) * n_points or i % n_points == 0 or i % n_points == n_points - 1:
            diags[0, i] = 1
            diags[1, min(i + 1, n_points ** 2 - 1)] = 0
            diags[2, max(i - 1, 0)] = 0
            diags[3, min(i + n_points, n_points ** 2 - 1)] = 0
            diags[4, max(i - n_points, 0)] = 0

    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, n_points, -n_points]), shape=(n_points ** 2, n_points ** 2)))


def dirichlet_bc(rhs, n_points, up, down, left, right):
    # filling of the four boundaries, inversion of up and down ?
    rhs[:n_points] = up
    rhs[n_points * (n_points - 1):] = down
    rhs[:n_points * (n_points - 1) + 1:n_points] = left
    rhs[n_points - 1::n_points] = right
    # mean approximation in case the potential is not continuous across boundaries
    rhs[0] = 0.5 * (up[0] + left[0])
    rhs[n_points - 1] = 0.5 * (up[-1] + right[1])
    rhs[-n_points] = 0.5 * (left[-1] + down[0])
    rhs[-1] = 0.5 * (right[-1] + down[-1])

def matrix_cart(dx, dy, nx, ny):
    diags = np.zeros((5, nx * ny))

    # Filling the diagonals, first the down neumann bc, then the dirichlet bc and finally the interior nodes
    for i in range(nx * ny):
        if 0 < i < nx - 1:
            diags[0, i] = - (2 / dx**2 + 1 / dy**2)
            diags[1, i + 1] = 1 / dx**2
            diags[2, i - 1] = 1 / dx**2
            diags[3, i + nx] = 1 / dy**2
        elif i >= (ny - 1) * nx or i % nx == 0 or i % nx == nx - 1:
            diags[0, i] = 1
            diags[1, min(i + 1, nx * ny - 1)] = 0
            diags[2, max(i - 1, 0)] = 0
            diags[3, min(i + nx, nx * ny - 1)] = 0
            diags[4, max(i - nx, 0)] = 0
        else:
            diags[0, i] = - (2 / dx**2 + 2 / dy**2)
            diags[1, i + 1] = 1 / dx**2
            diags[2, i - 1] = 1 / dx**2
            diags[3, i + nx] = 1 / dy**2
            diags[4, i - nx] = 1 / dy**2

    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, nx, -nx]), shape=(nx * ny, nx * ny)))

def matrix_axisym(dx, dr, nx, nr, R):
    diags = np.zeros((5, nx * nr))

    r = R.reshape(-1)

    # Filling the diagonals, first the down neumann bc, then the dirichlet bc and finally the interior nodes
    for i in range(nx * nr):
        if 0 < i < nx - 1:
            diags[0, i] = - ((r[i] + r[i + nx]) / (2 * r[i] * dr**2) + 2 / dx**2)
            diags[1, i + 1] = 1 / dx**2
            diags[2, i - 1] = 1 / dx**2
            diags[3, i + nx] = (r[i] + r[i + nx]) / (2 * r[i] * dr**2)
        elif i >= (nr - 1) * nx or i % nx == 0 or i % nx == nx - 1:
            diags[0, i] = 1
            diags[1, min(i + 1, nx * nr - 1)] = 0
            diags[2, max(i - 1, 0)] = 0
            diags[3, min(i + nx, nx * nr - 1)] = 0
            diags[4, max(i - nx, 0)] = 0
        else:
            diags[0, i] = - ((2 * r[i] + r[i + nx] + r[i - nx]) / (2 * r[i] * dr**2) + 2 / dx**2)
            diags[1, i + 1] = 1 / dx**2
            diags[2, i - 1] = 1 / dx**2
            diags[3, i + nx] = (r[i] + r[i + nx]) / (2 * r[i] * dr**2)
            diags[4, i - nx] = (r[i] + r[i - nx]) / (2 * r[i] * dr**2)


    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, nx, -nx]), shape=(nx * nr, nx * nr)))

def dirichlet_bc_axi(rhs, nx, nr, up, left, right):
    # filling of the three dirichlet boundaries for axisymmetric test case
    rhs[nx * (nr - 1):] = up
    rhs[:nx * (nr - 1) + 1:nx] = left
    rhs[nx - 1::nx] = right
    # mean approximation in case the potential is not continuous across boundaries
    rhs[-nx] = 0.5 * (left[-1] + up[0])
    rhs[-1] = 0.5 * (right[-1] + up[-1])