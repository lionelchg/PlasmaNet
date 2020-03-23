########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import scipy.constants as co
import time
from scipy.sparse.linalg import spsolve, inv
from poisson_setup_2D_FD import laplace_square_matrix, dirichlet_bc
from plot import plot_fig
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from scipy import interpolate
# Global variables
n_points = 64
xmin, xmax = 0, 0.01
ymin, ymax = 0, 0.01
dx = (xmax - xmin) / (n_points - 1)
dy = (ymax - ymin) / (n_points - 1)
x = np.linspace(xmin, xmax, n_points)
y = np.linspace(ymin, ymax, n_points)

n_lower = int(n_points / 4)
x_lower, y_lower = np.linspace(xmin, xmax, n_lower), np.linspace(ymin, ymax, n_lower)

L = xmax - xmin

X, Y = np.meshgrid(x, y)

A = laplace_square_matrix(n_points)

# Parameters for the rhs
ni0 = 1e16

def params(nits):
    for i in range(nits):
        z_lower = 2 * np.random.random((n_lower, n_lower)) - 1
        f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')
        yield f(x, y)

def compute(args):
    z = args
    #interior rhs
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

    plot = False
    n_procs = 36
    chunksize = 10

    nits = 10000
    print('Total number of inputs: %d' % nits)

    potential_random = np.zeros((nits, n_points, n_points))
    physical_rhs_random = np.zeros((nits, n_points, n_points))

    time_start = time.time()

    with Pool(processes=n_procs) as p:
        results_train = list(tqdm(p.imap(compute, params(nits), chunksize=chunksize), total=nits))

    for i, (pot, rhs) in enumerate(results_train):
        potential_random[i, :, :] = pot
        physical_rhs_random[i, :, :] = rhs
        if i % 10 == 0 and plot:
            plot_fig(X, Y, pot, rhs, name='random/dataset_1/intput_', nit=i)


    time_stop = time.time()
    np.save('datasets/rhs/%dx%d/potential_random.npy' % (n_points, n_points), potential_random)
    np.save('datasets/rhs/%dx%d/physical_rhs_random.npy' % (n_points, n_points), physical_rhs_random)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))