########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as co
from scipy.sparse.linalg import spsolve
from poissonsolver.operators import print_error, grad
from poissonsolver.plot import plot_set_2D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc

fig_dir = 'figures/superposition/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


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
    electric_field_0 = - grad(potential_0, dx, dy, n_points, n_points)
    physical_rhs_0 = physical_rhs.reshape(n_points, n_points)
    figname = fig_dir + 'rhs'
    plot_set_2D(X, Y, physical_rhs_0, potential_0, electric_field_0, 'RHS', figname)

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
    electric_field_dirichlet = - grad(potential_dirichlet, dx, dy, n_points, n_points)
    physical_rhs_dirichlet = physical_rhs.reshape(n_points, n_points)
    figname = fig_dir + 'dirichlet'
    plot_set_2D(X, Y, potential_dirichlet, potential_dirichlet, electric_field_dirichlet, 'Dirichlet', figname, no_rhs=True)

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
    electric_field = - grad(potential, dx, dy, n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    figname = fig_dir + 'full'
    plot_set_2D(X, Y, physical_rhs, potential, electric_field, 'Full',figname)

    potential_super = potential_0 + potential_dirichlet
    electric_field_super = - grad(potential_super, dx, dy, n_points, n_points)
    figname = fig_dir + 'superposition'
    plot_set_2D(X, Y, physical_rhs_0, potential_0, electric_field_0, 'Superposition', figname)

    print_error(potential_super, potential, dx*dy, Lx*Ly, "Error from superposition")

