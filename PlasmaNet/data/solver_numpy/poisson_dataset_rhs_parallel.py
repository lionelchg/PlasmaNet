########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import time
from multiprocessing import Pool

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
from .plot import plot_fig
from .poisson_setup_2D_FD import laplace_square_matrix, dirichlet_bc
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

# Global variables
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

# Parameters for the rhs
ni0 = 1e16
x0_min, x0_max = 4e-3, 6e-3
sigma_min, sigma_max = 1e-3, 3e-3
pow_min, pow_max = 3, 7


def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)


def cosine_hill(x, y, amplitude, x0, y0, powx, powy, L):
    return amplitude * np.cos(np.pi / L * (x - x0)) ** powx * np.cos(np.pi / L * (y - y0)) ** powy


def parabol(x, y, L):
    return (1 - ((x - L / 2) / (L / 2)) ** 2) * (1 - ((y - L / 2) / (L / 2)) ** 2)


def triangle(x, y, L):
    return (1 - abs((x - L / 2) / (L / 2))) * (1 - abs((y - L / 2) / (L / 2)))


def params_gauss(n_x0, n_sigma, n_points):
    x_range = np.linspace(x0_min, x0_max, n_x0)
    sigma_range = np.linspace(sigma_min, sigma_max, n_sigma)

    # Training set made out of gaussians
    for x0 in x_range:
        for y0 in x_range:
            for sigma_x in sigma_range:
                for sigma_y in sigma_range:
                    yield x0, y0, sigma_x, sigma_y, 0


def params_cosine(n_x0, n_sigma, n_points):
    x_range = np.linspace(x0_min, x0_max, n_x0)
    pow_range = np.linspace(pow_min, pow_max, n_pow)

    # Training set made out of gaussians
    for x0 in x_range:
        for y0 in x_range:
            for powx in pow_range:
                for powy in pow_range:
                    yield x0, y0, powx, powy, 1


def compute(args):
    x0, y0, param_x, param_y, which_set = args
    # interior rhs
    # Gaussian case
    if which_set == 0:
        physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, param_x, param_y) * co.e / co.epsilon_0
    # Cosine hill case
    elif which_set == 1:
        L = xmax - xmin
        physical_rhs = ni0 * co.e / co.epsilon_0 \
                       * cosine_hill(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, param_x, param_y,
                                     L) * co.e / co.epsilon_0
    rhs = - physical_rhs * dx ** 2

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

    # Solving the sparse linear system
    tmp_potential = spsolve(A, rhs).reshape(n_points, n_points)
    tmp_rhs = physical_rhs.reshape(n_points, n_points)

    return tmp_potential, tmp_rhs


if __name__ == '__main__':

    plot = False
    n_procs = 36
    chunksize = 10
    n_x0, n_sigma, n_pow = 25, 5, 3

    # test for sliding gaussian
    nit_gauss = n_x0 ** 2 * n_sigma ** 2
    nit_coshill = n_x0 ** 2 * n_pow ** 2
    nits = nit_gauss + nit_coshill
    print('Number of gauss inputs: %d' % nit_gauss)
    print('Number of coshill inputs: %d' % nit_coshill)
    print('Total number of inputs: %d' % nits)

    potential_gauss = np.zeros((nit_gauss, n_points, n_points))
    physical_rhs_gauss = np.zeros((nit_gauss, n_points, n_points))

    time_start = time.time()

    with Pool(processes=n_procs) as p:
        results_train = list(
            tqdm(p.imap(compute, params_gauss(n_x0, n_sigma, n_points), chunksize=chunksize), total=nit_gauss))

    for i, (pot, rhs) in enumerate(results_train):
        potential_gauss[i, :, :] = pot
        physical_rhs_gauss[i, :, :] = rhs
        if i % 50 == 0 and plot:
            plot_fig(X, Y, pot, rhs, name='hills/dataset_1/gauss_', nit=i)

    potential_coshill = np.zeros((nit_coshill, n_points, n_points))
    physical_rhs_coshill = np.zeros((nit_coshill, n_points, n_points))

    with Pool(processes=n_procs) as p:
        results_val = list(
            tqdm(p.imap(compute, params_cosine(n_x0, n_pow, n_points), chunksize=chunksize), total=nit_coshill))

    for i, (pot, rhs) in enumerate(results_val):
        potential_coshill[i, :, :] = pot
        physical_rhs_coshill[i, :, :] = rhs
        if i % 50 == 0 and plot:
            plot_fig(X, Y, pot, rhs, name='hills/dataset_1/coshill_', nit=i)

    time_stop = time.time()
    np.save('datasets/rhs/%dx%d/potential_gauss.npy' % (n_points, n_points), potential_gauss)
    np.save('datasets/rhs/%dx%d/physical_rhs_gauss.npy' % (n_points, n_points), physical_rhs_gauss)
    np.save('datasets/rhs/%dx%d/potential_coshill.npy' % (n_points, n_points), potential_coshill)
    np.save('datasets/rhs/%dx%d/physical_rhs_coshill.npy' % (n_points, n_points), physical_rhs_coshill)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
