########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import time
from multiprocessing import get_context

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from poissonsolver.plot import plot_set_2D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.operators import grad

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

# Directories declaration and creation if necessary
# data_dir = '/home/cfd/cheng/scratch/DL/datasets/rhs/random_%d_%d/' % (n_points, n_res)
data_dir = 'rhs/gauss_%d/' % n_points
fig_dir = data_dir + 'figures/'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

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
    n_procs = 36
    chunksize = 5
    n_ampl, n_x0, n_sigma, n_pow = 5, 10, 5, 5  # 5 * 12 * 12 * 5 * 5 = 18 000 for the gaussian set
    # n_ampl, n_x0, n_sigma, n_pow = 5, 4, 4, 3  # test dataset

    # test for sliding gaussian
    nits = n_ampl *  n_x0 ** 2 * n_sigma ** 2
    print('Total number of inputs: %d' % nits)

    potential_list = np.zeros((nits, n_points, n_points))
    physical_rhs_list = np.zeros((nits, n_points, n_points))

    time_start = time.time()

    with get_context('spawn').Pool(processes=n_procs) as p:
        results_train = list(tqdm(p.imap(compute, params_gauss(n_ampl, n_x0, n_sigma), chunksize=chunksize), total=nits))

    for i, (pot, rhs) in enumerate(tqdm(results_train)):
        potential_list[i, :, :] = pot
        physical_rhs_list[i, :, :] = rhs
        if i % 500 == 0 and plot:
            E_field = - grad(pot, dx, dy, n_points, n_points)
            plot_set_2D(X, Y, rhs, pot, E_field, 'Input number %d' % i, fig_dir + 'input_%d' % i)


    time_stop = time.time()
    np.save(data_dir + 'potential.npy', potential_list)
    np.save(data_dir + 'physical_rhs.npy', physical_rhs_list)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
