########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import scipy.constants as co
from scipy.sparse.linalg import spsolve

from operators import lapl, grad
from plot import plot_fig, plot_fig_scalar, plot_vector_arrow, plot_fig_list
from poisson_2D_FD import laplace_square_matrix, dirichlet_bc, lapl_diff


def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)


if __name__ == '__main__':
    n_points = 128
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    dx, dy = (xmax - xmin) / (n_points - 1), (ymax - ymin) / (n_points - 1)
    x, y = np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)

    X, Y = np.meshgrid(x, y)

    A = laplace_square_matrix(n_points)

    potential = np.zeros((n_points, n_points))
    physical_rhs = np.zeros((n_points, n_points))

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2
    rhs = np.zeros(n_points ** 2)

    # interior rhs
    physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    rhs = - physical_rhs * dx ** 2

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

    # Solving the sparse linear system
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    electric_field = grad(potential, dx, dy, n_points, n_points)
    field_energy = co.epsilon_0 / 2 * (electric_field[0]**2 + electric_field[1]**2)
    potential_energy = physical_rhs * co.epsilon_0 * potential
    interior_diff = lapl_diff(potential, physical_rhs, dx, dy, n_points, n_points)

    # Plots
    plot_fig(X, Y, potential, physical_rhs, name='tests/gauss_', nit=1)
    plot_fig_scalar(X, Y, interior_diff, 'Absolute difference',
                    'tests/gauss_abs_diff', colormap='Blues')
    plot_fig_list(X, Y, [field_energy, potential_energy, field_energy - potential_energy],
     ['Field Energy', 'Potential Energy', 'Difference'], 'tests/gauss_energies')
    plot_vector_arrow(X, Y, electric_field, "Electric field", "tests/gauss_electric_field")

