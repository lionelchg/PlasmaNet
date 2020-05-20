########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import sys
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

device = sys.argv[1]
n_points, n_ampl, n_x0, n_width = sys.argv[2:6]
n_points, n_ampl, n_x0, n_width = int(n_points), int(n_ampl), int(n_x0), int(n_width)

# Global variables
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
casename = f'{n_points:d}x{n_points:d}/gauss_val/'
if device == 'mac':
    data_dir = 'rhs/' + casename
    plot = True
    n_procs = 2
    chunksize = 20
    plot_period = 100
elif device == 'kraken':
    data_dir = '/scratch/cfd/cheng/DL/datasets/rhs/' + casename
    plot = True
    n_procs = 36
    chunksize = 5
    plot_period = 2000

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

fig_dir = data_dir + 'figures/'
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


def params_gauss(n_ampl, n_x0, n_width):
    for i in range(n_ampl):
        for j in range(n_x0**2):
            for k in range(n_width**2):
                coefs = np.random.random(5)
                yield (ampl_max - ampl_min) * coefs[0] + ampl_min, \
                      (x0_max - x0_min) * coefs[1] + x0_min, \
                      (x0_max - x0_min) * coefs[2] + x0_min, \
                      (sigma_max - sigma_min) * coefs[3] + sigma_min, \
                      (sigma_max - sigma_min) * coefs[4] + sigma_min, 0


def params_cosine(n_ampl, n_x0, n_pow):
    for i in range(n_ampl):
        for j in range(n_x0**2):
            for k in range(n_width**2):
                coefs = np.random.random(5)
                yield (ampl_max - ampl_min) * coefs[0] + ampl_min, \
                      (x0_max - x0_min) * coefs[1] + x0_min, \
                      (x0_max - x0_min) * coefs[2] + x0_min, \
                      (pow_max - pow_min) * coefs[3] + pow_min, \
                      (pow_max - pow_min) * coefs[4] + pow_min, 1


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

    nits = n_ampl *  n_x0 ** 2 * n_width ** 2
    # Print header of dataset
    print(f'Device : {device:s} - n_points = {n_points:d} - nits = {nits:d} (n_ampl={n_ampl:d}/n_x0={n_x0:d}/n_width={n_width:d})')
    print(f'Directory : {data_dir:s} - n_procs = {n_procs:d} - chunksize = {chunksize:d}')

    potential_list = np.zeros((nits, n_points, n_points))
    physical_rhs_list = np.zeros((nits, n_points, n_points))

    time_start = time.time()

    with get_context('spawn').Pool(processes=n_procs) as p:
        results_train = list(tqdm(p.imap(compute, params_gauss(n_ampl, n_x0, n_width), chunksize=chunksize), total=nits))

    for i, (pot, rhs) in enumerate(tqdm(results_train)):
        potential_list[i, :, :] = pot
        physical_rhs_list[i, :, :] = rhs
        if i % plot_period == 0 and plot:
            E_field = - grad(pot, dx, dy, n_points, n_points)
            plot_set_2D(X, Y, rhs, pot, E_field, 'Input number %d' % i, fig_dir + f'input_{i:05d}')


    time_stop = time.time()
    np.save(data_dir + 'potential.npy', potential_list)
    np.save(data_dir + 'physical_rhs.npy', physical_rhs_list)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
