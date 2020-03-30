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
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from plot import plot_fig
from poisson_setup_2D_FD import laplace_square_matrix, dirichlet_bc


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


def params_dirichlet(n1, n2):
    # Constant per branch up
    for comb in range(6):
        for i in range(0, n_points, n1):
            for j in range(0, n_points, n2):

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
                yield up, down, left, right


def compute(args):
    up, down, left, right = args

    rhs = np.zeros(n_points ** 2)

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, down, up, left, right)

    # Solving the sparse linear system
    tmp_potential = spsolve(A, rhs).reshape(n_points, n_points)

    return tmp_potential


if __name__ == '__main__':

    plot = False
    n_procs = 2
    chunksize = 40
    n1, n2 = 2, 2
    nrot = 1000

    # test for sliding gaussian
    nit_train = 6 * int(n_points / n1) * int(n_points / n2)
    nit_val = nrot
    nits = nit_train + nit_val
    print('Number of training inputs: %d' % nit_train)
    print('Number of validation inputs: %d' % nit_val)
    print('Total number of inputs: %d' % nits)
    potential = np.zeros((nits, n_points, n_points))

    time_start = time.time()

    with Pool(processes=n_procs) as p:
        results_train = list(tqdm(p.imap(compute, params_dirichlet(n1, n2), chunksize=chunksize), total=nit_train))

    for i, pot in enumerate(results_train):
        potential[i, :, :] = pot
        if i % 50 == 0 and plot:
            plot_fig(X, Y, pot, pot, name='dirichlet/dataset_1/train_', nit=i, no_rhs=True)

    with Pool(processes=n_procs) as p:
        results_val = list(tqdm(p.imap(compute, params_rotation(nrot), chunksize=chunksize), total=nit_val))

    for i, pot in enumerate(results_val):
        potential[i + nit_train, :, :] = pot
        if i % 50 == 0 and plot:
            plot_fig(X, Y, pot, pot, name='dirichlet/dataset_1/val_', nit=i, no_rhs=True)

    time_stop = time.time()
    np.save('datasets/dirichlet/%dx%d/potential.npy' % (n_points, n_points), potential)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
