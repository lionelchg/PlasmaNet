########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from scipy import sparse
import scipy.constants as co
from operators import lapl
import torch

def laplace_square_matrix(n_points):
    diags = np.zeros((5, n_points * n_points))

    # Filling the diagonal
    diags[0, :] = - 4 * np.ones(n_points ** 2)
    diags[1, :] = np.ones(n_points ** 2)
    diags[2, :] = np.ones(n_points ** 2)
    diags[3, :] = np.ones(n_points ** 2)
    diags[4, :] = np.ones(n_points ** 2)

    # Correcting the diagonal to take into account dirichlet boundary conditions
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
    # filling of the four boundaries
    rhs[:n_points] = up
    rhs[n_points * (n_points - 1):] = down
    rhs[:n_points * (n_points - 1) + 1:n_points] = left
    rhs[n_points - 1::n_points] = right
    # mean approximation in case the potential is not continuous across boundaries
    rhs[0] = 0.5 * (up[0] + left[0])
    rhs[n_points - 1] = 0.5 * (up[-1] + right[1])
    rhs[-n_points] = 0.5 * (left[-1] + down[0])
    rhs[-1] = 0.5 * (right[-1] + down[-1])

def lapl_diff(potential, physical_rhs, dx, dy, nx, ny):
    interior_diff = abs(lapl(potential, dx, dy, nx, ny) + physical_rhs)
    interior_diff[0, :] = 0
    interior_diff[-1, :] = 0
    interior_diff[:, 0] = 0
    interior_diff[:, -1] = 0
    return interior_diff

def func_energy(potential, electric_field, physical_rhs, voln):
    field_energy = 1 / 2 * (electric_field[0]**2 + electric_field[1]**2)
    potential_energy = physical_rhs * potential
    energy = np.sum((field_energy - potential_energy) * voln)
    return energy

def func_energy_torch(potential, electric_field, physical_rhs, voln):
    field_energy = 1 / 2 * (electric_field[0]**2 + electric_field[1]**2)
    potential_energy = physical_rhs * potential
    energy = torch.sum((field_energy - potential_energy) * voln)
    return energy

def compute_voln(X, dx, dy):
    voln = np.ones_like(X) * dx * dy
    voln[:, 0], voln[:, -1], voln[0, :], voln[-1, :] = \
        dx * dy / 2, dx * dy / 2, dx * dy / 2, dx * dy / 2
    voln[0, 0], voln[-1, 0], voln[0, -1], voln[-1, -1] = \
        dx * dy / 4, dx * dy / 4, dx * dy / 4, dx * dy / 4
    return voln