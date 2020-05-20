########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import sys
import os
from multiprocessing import get_context

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from scipy.sparse.linalg import spsolve
from poissonsolver.operators import lapl, grad
from poissonsolver.plot import plot_set_2D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.postproc import lapl_diff


device = sys.argv[1]
n_points, nits, N = sys.argv[2:5]
n_points, nits, N = int(n_points), int(nits), int(N)


xmin, xmax = 0, 0.01
ymin, ymax = 0, 0.01
Lx, Ly = xmax - xmin, ymax - ymin
dx, dy = (xmax - xmin) / (n_points - 1), (ymax - ymin) / (n_points - 1)
x, y = np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)

X, Y = np.meshgrid(x, y)

A = laplace_square_matrix(n_points)

potential = np.zeros((n_points, n_points))
physical_rhs = np.zeros((n_points, n_points))

# amplitude
ni0 = 1e16

# interior rhs
M = N
n_range, m_range = np.arange(1, N +1), np.arange(1, M + 1)
N_range, M_range = np.meshgrid(n_range, m_range)

casename = f'{n_points:d}x{n_points:d}/rand_fou_{N:d}_dec_train/'

if device == 'mac':
    data_dir = 'rhs/' + casename
    plot = True
    n_procs = 2
    chunksize = 20
    plot_period = 100
elif device == 'kraken':
    data_dir = '/scratch/cfd/cheng/DL/datasets/rhs/' + casename
    plot = False
    n_procs = 36
    chunksize = 5
    plot_period = 2000

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if plot:
    fig_dir = data_dir + 'figures/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def mode(X, Y, Lx, Ly, N, M):
    return np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly)

def sum_series(X, Y, Lx, Ly, coefs, N, M):
    series = np.zeros_like(X)
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            series += coefs[n - 1, m - 1] * np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly)
    return series

def pot_series(X, Y, Lx, Ly, coefs, N, M):
    series = np.zeros_like(X)
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            series += coefs[n - 1, m - 1] * np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly) / ((n * np.pi / Lx)**2 + (m * np.pi / Ly)**2)
    return series

def plot_mode_ampl(N_range, M_range, coefs, figname):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(N_range, M_range, coefs, alpha=0.7)
    ax.set_zlabel('Amplitude')
    ax.set_ylabel('M')
    ax.set_xlabel('N')
    ax.set_title('Mode amplitudes')
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')

def params(nits):
    for i in range(nits):
        random_array = np.random.random((N, M))
        rhs_coefs = ni0 * (2 * random_array - 1)
        yield rhs_coefs / (N_range**2 + M_range**2)

def compute(args):
    rhs_coefs = args
    # interior rhs
    tmp_potential = pot_series(X, Y, Lx, Ly, rhs_coefs, N, M)
    tmp_rhs = sum_series(X, Y, Lx, Ly, rhs_coefs, N, M)

    return tmp_potential, tmp_rhs, rhs_coefs

if __name__ == '__main__':

    # Print header of dataset
    print(f'Device : {device:s} - n_points = {n_points:d} - nits = {nits:d} - N (modes) = {N:d}')
    print(f'Directory : {data_dir:s} - n_procs = {n_procs:d} - chunksize = {chunksize:d}')

    potential_random = np.zeros((nits, n_points, n_points))
    physical_rhs_random = np.zeros((nits, n_points, n_points))

    time_start = time.time()

    with get_context('spawn').Pool(processes=n_procs) as p:
        results_train = list(tqdm(p.imap(compute, params(nits), chunksize=chunksize), total=nits))

    for i, (pot, rhs, coefs) in enumerate(tqdm(results_train)):
        potential_random[i, :, :] = pot
        physical_rhs_random[i, :, :] = rhs
        if i % plot_period == 0 and plot:
            E_field = - grad(potential_random[i, :, :], dx, dy, n_points, n_points)
            plot_set_2D(X, Y, physical_rhs_random[i, :, :], potential_random[i, :, :], E_field, f'Fourier random {i:d}', fig_dir + f'input_{i:05d}')
            plot_mode_ampl(N_range, M_range, coefs, fig_dir + f'input_{i:05d}_mode_ampl')

    time_stop = time.time()
    np.save(data_dir + 'potential.npy', potential_random)
    np.save(data_dir + 'physical_rhs.npy', physical_rhs_random)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
