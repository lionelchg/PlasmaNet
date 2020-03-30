########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from scipy import interpolate
from scipy.sparse.linalg import spsolve

from plot import plot_fig
from poisson_setup_2D_FD import laplace_square_matrix, dirichlet_bc

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

    rhs = np.zeros(n_points**2)
    physical_rhs = np.zeros(n_points**2)

    V = 100
    x_1D = np.linspace(xmin, xmax, n_points)
    xm = 0.5 * (xmin + xmax)
    L = xmax - xmin
    ones_bc = np.ones(n_points)
    zeros_bc = np.zeros(n_points)
    linear_bc = np.linspace(0, V, n_points)

    n_lower = int(n_points / 4)
    x_1D_lower = np.linspace(xmin, xmax, n_lower)
    random_1D = V * (2 * np.random.random(n_lower) - 1)
    f = interpolate.interp1d(x_1D_lower, random_1D)

    # Linear potential
    up = zeros_bc
    down = zeros_bc
    left = f(x_1D)
    right = zeros_bc

    dirichlet_bc(rhs, n_points, down, up, left, right)
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    plot_fig(X, Y, potential, physical_rhs.reshape(n_points, n_points), name='dirichlet/random_', nit=0, no_rhs=True)

