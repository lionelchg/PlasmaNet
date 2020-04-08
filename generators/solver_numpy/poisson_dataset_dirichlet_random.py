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
from scipy import interpolate
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from plot import plot_fig
from poisson_2D_FD import laplace_square_matrix, dirichlet_bc


# Global variables
n_points = 64
n_res = 4
n_lower = int(n_points / n_res)
xmin, xmax = 0, 0.01
ymin, ymax = 0, 0.01
dx = (xmax - xmin) / (n_points - 1)
dy = (ymax - ymin) / (n_points - 1)
x = np.linspace(xmin, xmax, n_points)
y = np.linspace(ymin, ymax, n_points)

x_1D_lower = np.linspace(xmin, xmax, n_lower)

L = xmax - xmin

X, Y = np.meshgrid(x, y)

A = laplace_square_matrix(n_points)

# potential value for the test cases
V = 100
ones_bc = np.ones(n_points)
zeros_bc = np.zeros(n_points)
linear_bc = np.linspace(0, V, n_points)
linear_xy = x


def params_rotation(nrot):
    # rotated linear potential

    thetas = np.linspace(0, 2 * np.pi, nrot)
    for index, theta in enumerate(thetas):
        up = V * (np.cos(theta) * linear_xy - np.sin(theta) * ymax)
        down = V * np.cos(theta) * linear_xy
        left = - V * np.sin(theta) * linear_xy
        right = V * (np.cos(theta) * xmax - np.sin(theta) * linear_xy)

        yield up, down, left, right


def params_random(nits):
    # Constant per branch up
    up = zeros_bc
    down = zeros_bc
    right = zeros_bc
    for i in range(nits):
        random_1D = V * (2 * np.random.random(n_lower) - 1)
        f = interpolate.interp1d(x_1D_lower, random_1D)

        # Linear potential
        left = f(x)
        yield up, down, left, right


def compute(args):
    up, down, left, right = args

    rhs = np.zeros(n_points ** 2)

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, down, up, left, right)

    # Solving the sparse linear system
    tmp_potential = spsolve(A, rhs).reshape(n_points, n_points)

    return tmp_potential, left


if __name__ == '__main__':

    dataset_dir = 'datasets/dirichlet/random_%d_%d/' % (n_points, n_res)
    fig_dir = 'figures/' + dataset_dir

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    plot = True
    n_procs = 2
    chunksize = 40

    # test for sliding gaussian
    nits = 500
    print('Total number of inputs: %d' % nits)
    potential = np.zeros((nits, n_points, n_points))
    potential_boundary = np.zeros((nits, n_points))

    time_start = time.time()

    with Pool(processes=n_procs) as p:
        results = list(tqdm(p.imap(compute, params_random(nits), chunksize=chunksize), total=nits))

    for i, result in enumerate(results):
        potential[i, :, :] = result[0]
        potential_boundary[i, :] = result[1]
        if i % 50 == 0 and plot:
            plot_fig(X, Y, result[0], result[0], name=dataset_dir + 'input_', nit=i, no_rhs=True)


    time_stop = time.time()
    np.save(dataset_dir + 'potential.npy', potential)
    np.save(dataset_dir + 'potential_boundary.npy', potential_boundary)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
