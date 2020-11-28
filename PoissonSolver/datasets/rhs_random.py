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
from scipy import interpolate
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from poissonsolver.plot import plot_set_2D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.operators import grad

device = sys.argv[1]
n_points, nits, n_res = sys.argv[2:5]
n_points, nits, n_res = int(n_points), int(nits), int(n_res)

# Global variables
xmin, xmax = 0, 0.01
ymin, ymax = 0, 0.01
dx = (xmax - xmin) / (n_points - 1)
dy = (ymax - ymin) / (n_points - 1)
x = np.linspace(xmin, xmax, n_points)
y = np.linspace(ymin, ymax, n_points)

n_lower = int(n_points / n_res)
x_lower, y_lower = np.linspace(xmin, xmax, n_lower), np.linspace(ymin, ymax, n_lower)

L = xmax - xmin

X, Y = np.meshgrid(x, y)

A = laplace_square_matrix(n_points)

# Parameters for the rhs
ni0 = 1e11

# Directories declaration and creation if necessary
casename = f'{n_points:d}x{n_points}/random_{n_res:d}/'
if device == 'mac':
    data_dir = 'rhs/' + casename
    plot = True
    n_procs = 2
    chunksize = 20
    plot_period = 100
elif device == 'kraken':
    data_dir = '/scratch/cfd/cheng/DL/datasets/' + casename
    plot = True
    n_procs = int(sys.argv[5])
    chunksize = 5
    plot_period = 1000

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if plot:
    fig_dir = data_dir + 'figures/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)


def params(nits):
    for i in range(nits):
        z_lower = 2 * np.random.random((n_lower, n_lower)) - 1
        f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')
        yield f(x, y)


def compute(args):
    z = args
    # interior rhs
    physical_rhs = ni0 * z.reshape(-1) * co.e / co.epsilon_0

    rhs = - physical_rhs * dx**2

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

    # Solving the sparse linear system
    tmp_potential = spsolve(A, rhs).reshape(n_points, n_points)
    tmp_rhs = physical_rhs.reshape(n_points, n_points)

    return tmp_potential, tmp_rhs


if __name__ == '__main__':

    # Print header of dataset
    print(f'Device : {device:s} - n_points = {n_points:d} - nits = {nits:d} - n_res = {n_res:d}')
    print(f'Directory : {data_dir:s} - n_procs = {n_procs:d} - chunksize = {chunksize:d}')

    potential_random = np.zeros((nits, n_points, n_points))
    physical_rhs_random = np.zeros((nits, n_points, n_points))

    time_start = time.time()

    with get_context('spawn').Pool(processes=n_procs) as p:
        results_train = list(tqdm(p.imap(compute, params(nits), chunksize=chunksize), total=nits))

    for i, (pot, rhs) in enumerate(tqdm(results_train)):
        potential_random[i, :, :] = pot
        physical_rhs_random[i, :, :] = rhs
        if i % plot_period == 0 and plot:
            E_field = - grad(potential_random[i, :, :], dx, dy, n_points, n_points)
            plot_set_2D(X, Y, physical_rhs_random[i, :, :], potential_random[i, :, :], E_field, f'Random {i:d}', fig_dir + f'input_{i:05d}')


    time_stop = time.time()
    np.save(data_dir + 'potential.npy', potential_random)
    np.save(data_dir + 'physical_rhs.npy', physical_rhs_random)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
