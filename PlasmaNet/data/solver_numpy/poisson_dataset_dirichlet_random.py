########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import scipy.constants as co
import time
from scipy.sparse.linalg import spsolve, inv
from poisson_setup_2D_FD import laplace_square_matrix, dirichlet_bc
from plot import plot_fig
import os
from multiprocessing import Pool
from tqdm import tqdm
from scipy import interpolate

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Global variables
n_points = 64
xmin, xmax = 0, 0.01
ymin, ymax = 0, 0.01
dx = (xmax - xmin) / (n_points - 1)
dy = (ymax - ymin) / (n_points - 1)
x = np.linspace(xmin, xmax, n_points)
y = np.linspace(ymin, ymax, n_points)

n_lower = int(n_points / 4)
x_lower = np.linspace(xmin, xmax, n_lower)

L = xmax - xmin

X, Y = np.meshgrid(x, y)

A = laplace_square_matrix(n_points)

# potential value for the test cases
V = 100
ones_bc = np.ones(n_points)
zeros_bc = np.zeros(n_points)
linear_bc = np.linspace(0, V, n_points)
linear_xy = x

def params_dirichlet(nits):
    for i in range(nits):
        random_1D = V * (2 * np.random.random(n_lower) - 1)
        f = interpolate.interp1d(x_lower, random_1D, kind='cubic')
        yield f(x)


def compute(args):
    left = args
    up, down, right = zeros_bc, zeros_bc, zeros_bc
    rhs = np.zeros(n_points**2)

    # Imposing Dirichlet boundary conditions
    dirichlet_bc(rhs, n_points, down, up, left.reshape(-1), right)

    # Solving the sparse linear system
    tmp_potential = spsolve(A, rhs).reshape(n_points, n_points)

    return tmp_potential

if __name__ == '__main__':

    plot = True
    n_procs = 2
    chunksize = 40

    nits = 500
    print('Total number of inputs: %d' % nits)
    potential = np.zeros((nits, n_points, n_points))

    time_start = time.time()

    with Pool(processes=n_procs) as p:
        results_train = list(tqdm(p.imap(compute, params_dirichlet(nits), chunksize=chunksize), total=nits))

    for i, pot in enumerate(results_train):
        potential[i, :, :] = pot
        if i % 50 == 0 and plot:
            plot_fig(X, Y, pot, pot, name='dirichlet/dataset_1/random_', nit=i, no_rhs=True)


    time_stop = time.time()
    np.save('datasets/dirichlet/%dx%d/potential.npy' % (n_points, n_points), potential)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
