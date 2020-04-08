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
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from plot import plot_fig
from poisson_2D_FD import laplace_square_matrix, dirichlet_bc

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

# Path variables
case_name = 'ampl'
data_dir = Path('./datasets')
save_dir = data_dir / '{0}x{0}'.format(n_points) / 'rhs'
save_dir_gauss = save_dir / 'gauss_{}'.format(case_name)
save_dir_coshill = save_dir / 'coshill_{}'.format(case_name)
save_dir_gauss.mkdir(parents=True, exist_ok=True)
save_dir_coshill.mkdir(parents=True, exist_ok=True)

# Parameters for the rhs
# ni0 = 1e16
ampl_min, ampl_max = 1e16, 5e16
# ampl_min, ampl_max = -0.5e16, 7.5e16  # test dataset
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


def params_gauss(n_ampl, n_x0, n_sigma):
    ampl_range = np.linspace(ampl_min, ampl_max, n_ampl)
    x_range = np.linspace(x0_min, x0_max, n_x0)
    sigma_range = np.linspace(sigma_min, sigma_max, n_sigma)

    # Training set made out of gaussians
    for ampl in ampl_range:
        for x0 in x_range:
            for y0 in x_range:
                for sigma_x in sigma_range:
                    for sigma_y in sigma_range:
                        yield ampl, x0, y0, sigma_x, sigma_y, 0


def params_cosine(n_ampl, n_x0, n_sigma):
    ampl_range = np.linspace(ampl_min, ampl_max, n_ampl)
    x_range = np.linspace(x0_min, x0_max, n_x0)
    pow_range = np.linspace(pow_min, pow_max, n_pow)

    # Training set made out of cosines
    for ampl in ampl_range:
        for x0 in x_range:
            for y0 in x_range:
                for powx in pow_range:
                    for powy in pow_range:
                        yield ampl, x0, y0, powx, powy, 1


def compute(args):
    ampl, x0, y0, param_x, param_y, which_set = args
    # interior rhs
    # Gaussian case
    if which_set == 0:
        physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ampl, x0, y0, param_x, param_y) * co.e / co.epsilon_0
    # Cosine hill case
    elif which_set == 1:
        L = xmax - xmin
        physical_rhs = ampl * co.e / co.epsilon_0 \
                       * cosine_hill(X.reshape(-1), Y.reshape(-1), ampl, x0, y0, param_x, param_y,
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

    plot = True
    n_procs = 2
    chunksize = 10
    n_ampl, n_x0, n_sigma, n_pow = 1, 3, 3, 3  # 5 * 12 * 12 * 5 * 5 = 18 000 for the gaussian set
    # n_ampl, n_x0, n_sigma, n_pow = 5, 4, 4, 3  # test dataset

    # test for sliding gaussian
    nit_gauss = n_ampl *  n_x0 ** 2 * n_sigma ** 2
    nit_coshill = n_ampl * n_x0 ** 2 * n_pow ** 2
    nits = nit_gauss + nit_coshill
    print('Number of gauss inputs: %d' % nit_gauss)
    print('Number of coshill inputs: %d' % nit_coshill)
    print('Total number of inputs: %d' % nits)

    if plot:
        os.makedirs('figures/dataset_{}'.format(case_name), exist_ok=True)

    potential_gauss = np.zeros((nit_gauss, n_points, n_points))
    physical_rhs_gauss = np.zeros((nit_gauss, n_points, n_points))
    potential_coshill = np.zeros((nit_coshill, n_points, n_points))
    physical_rhs_coshill = np.zeros((nit_coshill, n_points, n_points))

    time_start = time.time()

    with Pool(processes=n_procs) as p:
        results_train = list(
            tqdm(p.imap(compute, params_gauss(n_ampl, n_x0, n_sigma), chunksize=chunksize), total=nit_gauss,
                 desc='Compute gauss'))
        results_val = list(
            tqdm(p.imap(compute, params_cosine(n_ampl, n_x0, n_pow), chunksize=chunksize), total=nit_coshill,
                 desc='Compute coshill'))

    for i, (pot, rhs) in tqdm(enumerate(results_train), total=nit_gauss, desc='Save gauss'):
        potential_gauss[i, :, :] = pot
        physical_rhs_gauss[i, :, :] = rhs
        if plot and i % 10 == 0:
            plot_fig(X, Y, pot, rhs, name='datasets/rhs/gauss/input_', nit=i)

    for i, (pot, rhs) in tqdm(enumerate(results_val), total=nit_coshill, desc='Save coshill'):
        potential_coshill[i, :, :] = pot
        physical_rhs_coshill[i, :, :] = rhs
        if plot and i % 10 == 0:
            plot_fig(X, Y, pot, rhs, name='datasets/rhs/coshill/input_', nit=i)

    time_stop = time.time()
    np.save(save_dir_gauss / 'potential.npy', potential_gauss)
    np.save(save_dir_gauss / 'physical_rhs.npy', physical_rhs_gauss)
    np.save(save_dir_coshill / 'potential.npy', potential_coshill)
    np.save(save_dir_coshill / 'physical_rhs.npy', physical_rhs_coshill)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
