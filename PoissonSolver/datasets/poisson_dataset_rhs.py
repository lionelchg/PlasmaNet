########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import time

import numpy as np
import scipy.constants as co
from scipy.sparse.linalg import spsolve

from plot import plot_fig
from poisson_2D_FD import laplace_square_matrix, dirichlet_bc


def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)


def cosine_hill(x, y, amplitude, x0, y0, powx, powy, L):
    return amplitude * np.cos(np.pi / L * (x - x0)) ** powx * np.cos(np.pi / L * (y - y0)) ** powy


def parabol(x, y, L):
    return (1 - ((x - L / 2) / (L / 2)) ** 2) * (1 - ((y - L / 2) / (L / 2)) ** 2)


def triangle(x, y, L):
    return (1 - abs((x - L / 2) / (L / 2))) * (1 - abs((y - L / 2) / (L / 2)))


if __name__ == '__main__':

    plot = False

    n_points = 64
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    dx = (xmax - xmin) / (n_points - 1)
    dy = (ymax - ymin) / (n_points - 1)
    x = np.linspace(xmin, xmax, n_points)
    y = np.linspace(ymin, ymax, n_points)

    L = xmax - xmin

    X, Y = np.meshgrid(x, y)

    A = laplace_square_matrix(n_points)

    time_start = time.time()
    ni0 = 1e16
    n_x0, n_sigma, n_pow = 21, 5, 3
    x_range = np.linspace(4e-3, 6e-3, n_x0)
    sigma_range = np.linspace(1e-3, 3e-3, n_sigma)
    pow_range = np.linspace(3, 5, n_pow)
    nit = 0

    # test for sliding gaussian
    nits = n_x0 ** 2 * n_sigma ** 2 + n_x0 ** 2 * n_pow ** 2
    potential = np.zeros((nits, n_points, n_points))
    physical_rhs_list = np.zeros((nits, n_points, n_points))

    # Training set made out of gaussians
    for x0 in x_range:
        for y0 in x_range:
            for sigma_x in sigma_range:
                for sigma_y in sigma_range:
                    # creating the rhs
                    rhs = np.zeros(n_points ** 2)

                    # interior rhs
                    physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x,
                                            sigma_y) * co.e / co.epsilon_0
                    rhs = - physical_rhs * dx ** 2

                    # Imposing Dirichlet boundary conditions
                    zeros_bc = np.zeros(n_points)
                    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

                    # Solving the sparse linear system
                    potential[nit, :, :] = spsolve(A, rhs).reshape(n_points, n_points)
                    physical_rhs_list[nit, :, :] = physical_rhs.reshape(n_points, n_points)
                    if nit % 50 == 0 and plot:
                        plot_fig(X, Y, potential[nit, :, :], physical_rhs_list[nit, :, :], name='gauss/potential_2D_',
                                 nit=nit)
                    nit += 1
                    if nit % 200 == 0:
                        print('nit = %d' % nit)

    nit_train = nit
    print('Number of training inputs: %d' % nit_train)

    # Validation set made out of cosine hills
    for x0 in x_range:
        for y0 in x_range:
            for powx in pow_range:
                for powy in pow_range:
                    # creating the rhs
                    rhs = np.zeros(n_points ** 2)

                    # interior rhs
                    # physical_rhs = parabol(X.reshape(-1), Y.reshape(-1), L) * cosine_hill(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, powx, powy, L) * co.e / co.epsilon_0
                    physical_rhs = ni0 * parabol(X.reshape(-1), Y.reshape(-1), L) * co.e / co.epsilon_0 * cosine_hill(
                        X.reshape(-1), Y.reshape(-1), ni0, x0, y0, powx, powy, L) * co.e / co.epsilon_0
                    rhs = - physical_rhs * dx ** 2

                    # Imposing Dirichlet boundary conditions
                    zeros_bc = np.zeros(n_points)
                    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

                    # Solving the sparse linear system
                    potential[nit, :, :] = spsolve(A, rhs).reshape(n_points, n_points)
                    physical_rhs_list[nit, :, :] = physical_rhs.reshape(n_points, n_points)
                    if nit % 50 == 0 and plot:
                        plot_fig(X, Y, potential[nit, :, :], physical_rhs_list[nit, :, :], name='gauss/potential_2D_',
                                 nit=nit)
                    nit += 1
                    if nit % 200 == 0:
                        print('nit = %d' % nit)

    print('Number of validation inputs: %d' % (nit - nit_train))

    time_stop = time.time()
    np.save('datasets/potential.npy', potential)
    np.save('datasets/physical_rhs.npy', physical_rhs_list)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
