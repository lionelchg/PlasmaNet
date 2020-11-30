########################################################################################################################
#                                                                                                                      #
#                              2D Poisson datasets using random generation of rhs gaussian                             #
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
from poissonsolver.operators import grad
from poissonsolver.poisson import DatasetPoisson
from poissonsolver.utils import create_dir
from poissonsolver.funcs import gaussian

device = sys.argv[1]
npts, nits, n_procs = [int(var) for var in sys.argv[2:5]]

xmin, xmax, nnx = 0, 0.01, npts
ymin, ymax, nny = 0, 0.01, npts
x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)

poisson = DatasetPoisson(xmin, xmax, nnx, ymin, ymax, nny, 'cart_dirichlet', 10)

# Parameters for the rhs and plotting
ni0 = 1e11
rhs0 = ni0 * co.e / co.epsilon_0
ampl_min, ampl_max = 0.01, 1
sigma_min, sigma_max = 1e-3, 3e-3
x_middle, y_middle = (xmax - xmin) / 2, (ymax - ymin) / 2

plot = True
plot_period = int(0.1 * nits)
freq_period = int(0.01 * nits)

# Directories declaration and creation if necessary
casename = f'{npts:d}x{npts}/hills/'
if device == 'mac':
    data_dir = 'outputs/' + casename
    chunksize = 20
elif device == 'kraken':
    data_dir = '/scratch/cfd/cheng/DL/datasets/' + casename
    chunksize = 5

fig_dir = data_dir + 'figures/'
create_dir(data_dir)
create_dir(fig_dir)


def params(nits):
    """ Parameters to give to compute function for imap """
    for i in range(nits):
        coefs = np.random.random(5)
        ampl = rhs0 * ((ampl_max - ampl_min) * coefs[0] + ampl_min)
        sigma0 = (sigma_max - sigma_min) * coefs[1] + sigma_min
        yield ampl, sigma0


def compute(args):
    """ Compute function for imap (multiprocessing) """
    ampl, sigma0 = args
    physical_rhs = gaussian(poisson.X.reshape(-1), poisson.Y.reshape(-1), ampl, 
                        x_middle, y_middle, sigma0, sigma0)

    poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)

    return poisson.potential, poisson.physical_rhs

if __name__ == '__main__':

    # Print header of dataset
    print(f'Device : {device:s} - npts = {npts:d} - nits = {nits:d}')
    print(f'Directory : {data_dir:s} - n_procs = {n_procs:d} - chunksize = {chunksize:d}')

    potential_list = np.zeros((nits, npts, npts))
    physical_rhs_list = np.zeros((nits, npts, npts))

    time_start = time.time()

    with get_context('spawn').Pool(processes=n_procs) as p:
        results_train = list(tqdm(p.imap(compute, params(nits), chunksize=chunksize), total=nits))

    for i, (pot, rhs) in enumerate(tqdm(results_train)):
        potential_list[i, :, :] = pot
        physical_rhs_list[i, :, :] = rhs
        if i % plot_period == 0:
            poisson.potential = pot
            poisson.plot_2D(fig_dir + f'input_{i:05d}')
        if i % freq_period == 0:
            poisson.physical_rhs = rhs
            poisson.compute_modes()
    
    poisson.plot_pmodes(fig_dir + 'average_modes')

    time_stop = time.time()
    np.save(data_dir + 'potential.npy', potential_list)
    np.save(data_dir + 'physical_rhs.npy', physical_rhs_list)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
