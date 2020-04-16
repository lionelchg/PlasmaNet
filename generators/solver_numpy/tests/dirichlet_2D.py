########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import numpy as np
from scipy.sparse.linalg import spsolve
from poissonsolver.plot import plot_set_2D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.operators import grad

# Creation of directories
fig_dir = 'figures/dirichlet_2D/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)


if __name__ == '__main__':

    plot = True

    n_points = 64
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    dx = (xmax - xmin) / (n_points - 1)
    dy = (ymax - ymin) / (n_points - 1)
    x = np.linspace(xmin, xmax, n_points)
    y = np.linspace(ymin, ymax, n_points)

    X, Y = np.meshgrid(x, y)

    A = laplace_square_matrix(n_points)

    # test for dirichlet boundary conditions only (no rhs)
    potential = np.zeros((n_points, n_points))

    rhs = np.zeros(n_points ** 2)
    physical_rhs = np.zeros(n_points ** 2)

    V = 100
    linear_xy = np.linspace(xmin, xmax, n_points)
    xm = 0.5 * (xmin + xmax)
    L = xmax - xmin
    ones_bc = np.ones(n_points)
    zeros_bc = np.zeros(n_points)
    linear_bc = np.linspace(0, V, n_points)

    # Linear potential
    up = linear_bc
    down = linear_bc
    left = zeros_bc
    right = V * ones_bc

    dirichlet_bc(rhs, n_points, down, up, left, right)
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    electric_field = - grad(potential, dx, dy, n_points, n_points)

    casename = 'linear_potential'
    figname = fig_dir + casename
    # Plots
    plot_set_2D(X, Y, physical_rhs, potential, electric_field, 'Linear potential', figname, no_rhs=True)


    # Constant up
    up = V * ones_bc
    down = zeros_bc
    left = zeros_bc
    right = zeros_bc

    dirichlet_bc(rhs, n_points, down, up, left, right)
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    electric_field = - grad(potential, dx, dy, n_points, n_points)

    casename = 'constant_up'
    figname = fig_dir + casename
    plot_set_2D(X, Y, physical_rhs, potential, electric_field, 'Potential up', figname, no_rhs=True)