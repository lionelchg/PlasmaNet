########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as co

from .operators import lapl, print_error
from .plot import plot_fig, plot_ax
from .poisson_setup_2D_FD import laplace_square_matrix, dirichlet_bc
from scipy.sparse.linalg import spsolve


def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)


if __name__ == '__main__':
    n_points = 128
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    Lx, Ly = xmax - xmin, ymax - ymin
    dx, dy = (xmax - xmin) / (n_points - 1), (ymax - ymin) / (n_points - 1)
    x, y = np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)

    X, Y = np.meshgrid(x, y)

    A = laplace_square_matrix(n_points)

    potential = np.zeros((n_points, n_points))
    physical_rhs = np.zeros((n_points, n_points))

    #######################
    # rhs, zero dirichlet #
    #######################

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.8e-2, 0.8e-2
    rhs = np.zeros(n_points ** 2)

    # ZERO DIRICHLET
    # interior rhs
    physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    rhs = - physical_rhs * dx ** 2

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

    # Solving the sparse linear system
    potential_0 = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs_0 = physical_rhs.reshape(n_points, n_points)
    plot_fig(X, Y, potential_0, physical_rhs_0, name='superposition/potential_', nit=0)

    ##################
    # dirichlet only #
    ##################

    # interior rhs
    physical_rhs = np.zeros(n_points ** 2)
    rhs = - physical_rhs * dx ** 2

    # dirichlet boundary conditions
    V = 1000
    ones_bc = np.ones(n_points)
    linear_bc = np.linspace(0, V, n_points)
    down = linear_bc
    up = linear_bc
    left = zeros_bc
    right = V * ones_bc

    dirichlet_bc(rhs, n_points, down, up, left, right)
    potential_dirichlet = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs_dirichlet = physical_rhs.reshape(n_points, n_points)
    plot_fig(X, Y, potential_dirichlet, physical_rhs_dirichlet, name='superposition/potential_', nit=1)

    ################
    # full problem #
    ################

    # interior rhs
    physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    rhs = - physical_rhs * dx ** 2

    # dirichlet boundary conditions
    V = 1000
    ones_bc = np.ones(n_points)
    linear_bc = np.linspace(0, V, n_points)
    down = linear_bc
    up = linear_bc
    left = zeros_bc
    right = V * ones_bc

    dirichlet_bc(rhs, n_points, down, up, left, right)
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    plot_fig(X, Y, potential, physical_rhs, name='superposition/potential_', nit=2)

    potential_super = potential_0 + potential_dirichlet
    plot_fig(X, Y, potential_super, physical_rhs.reshape(n_points, n_points), name='superposition/potential_', nit=3)

    print_error(potential_super, potential, dx*dy, Lx*Ly, "Error from superposition")

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15))
    levels = plot_ax(fig, axes[0], X, Y, potential_0, physical_rhs_0, npot=1)
    plot_ax(fig, axes[1], X, Y, potential_dirichlet, physical_rhs_dirichlet, levels=levels, npot=2)
    plot_ax(fig, axes[2], X, Y, potential, physical_rhs)
    plt.savefig('figures/superposition/superposition', bbox_inches='tight')
