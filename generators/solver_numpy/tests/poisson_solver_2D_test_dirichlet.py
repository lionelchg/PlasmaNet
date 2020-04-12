########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from scipy.sparse.linalg import spsolve

from plot import plot_fig
from poisson_2D_FD import laplace_square_matrix, dirichlet_bc


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
    plot_fig(X, Y, potential, physical_rhs.reshape(n_points, n_points), name='dirichlet/linear_', nit=0, no_rhs=True)

    # rotated linear potential

    thetas = np.linspace(-np.pi / 2, np.pi / 2, 20)
    for index, theta in enumerate(thetas):
        up = V * (np.cos(theta) * linear_xy - np.sin(theta) * ymax)
        down = V * np.cos(theta) * linear_xy
        left = - V * np.sin(theta) * linear_xy
        right = V * (np.cos(theta) * xmax - np.sin(theta) * linear_xy)
        dirichlet_bc(rhs, n_points, down, up, left, right)
        potential = spsolve(A, rhs).reshape(n_points, n_points)
        if index % 2 == 0:
            plot_fig(X, Y, potential, physical_rhs.reshape(n_points, n_points), name='dirichlet/linear_rot_', nit=index,
                     no_rhs=True)

    # Constant up
    up = V * ones_bc
    down = zeros_bc
    left = zeros_bc
    right = zeros_bc

    dirichlet_bc(rhs, n_points, down, up, left, right)
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    plot_fig(X, Y, potential / V, physical_rhs.reshape(n_points, n_points), name='dirichlet/V_up_', nit=0, no_rhs=True)

    # Constant per branch up
    tot = 0
    for comb in range(6):
        for i in range(0, n_points, 16):
            for j in range(0, n_points, 16):

                up = np.zeros(n_points)
                down = np.zeros(n_points)
                left = np.zeros(n_points)
                right = np.zeros(n_points)

                if comb == 0:
                    down[:i] = V
                    left[:j] = V
                elif comb == 1:
                    left[:i] = V
                    up[:j] = V
                elif comb == 2:
                    up[:i] = V
                    right[:j] = V
                elif comb == 3:
                    right[:i] = V
                    down[:j] = V
                elif comb == 4:
                    down[:i] = V
                    up[:j] = V
                elif comb == 5:
                    left[:i] = V
                    right[:j] = V

                dirichlet_bc(rhs, n_points, down, up, left, right)
                potential = spsolve(A, rhs).reshape(n_points, n_points)
                if tot % 10 == 0:
                    plot_fig(X, Y, potential / V, physical_rhs.reshape(n_points, n_points), name='dirichlet/test_',
                             nit=tot, no_rhs=True)
                tot += 1
