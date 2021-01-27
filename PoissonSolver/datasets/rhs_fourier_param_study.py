########################################################################################################################
#                                                                                                                      #
#                              2D Poisson datasets using random generation of rhs points                               #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import sys
import os
import time
from multiprocessing import get_context
import argparse

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

args = argparse.ArgumentParser(description='Rhs random dataset')
args.add_argument('-c', '--casename', default=None, type=str,
                    help='dataset name')
args.add_argument('-d', '--device', default=None, type=str,
                    help='device on which the dataset is run')
args.add_argument('-n', '--npts', default=101, type=int,
                    help='number of points of the domain')
args.add_argument('-ni', '--nits', default=None, type=int,
                    help='number of entries in the dataset')
args.add_argument('-nm', '--nmax_fourier', default=None, type=int,
                    help='number of fourier modes included')
args.add_argument('-np', '--n_procs', default=None, type=int,
                    help='number of procs')
args.add_argument('-uf', '--unfilmod', default=None, type=int,
                    help='dominant mode on a unimodal distribution')
args = args.parse_args()

device = args.device
npts, nits, nmax_fourier, n_procs = args.npts, args.nits, args.nmax_fourier, args.n_procs
unfiltered_mode = args.unfilmod
casename = args.casename

xmin, xmax, nnx = 0, 0.01, npts
ymin, ymax, nny = 0, 0.01, npts
x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)

poisson = DatasetPoisson(xmin, xmax, nnx, ymin, ymax, nny, 'cart_dirichlet', nmax_fourier)

# amplitude of the random modes
ni0 = 1e11
rhs0 = ni0 * co.e / co.epsilon_0

# Parameters for the plotting
plot = True
plot_period = int(0.1 * nits)
freq_period = int(0.01 * nits)

# Directories declaration and creation if necessary
casename_l = 'fourier_single_mode/'.format(npts, npts, unfiltered_mode) + casename +'/'
if device == 'mac':
    data_dir = 'outputs/' + casename_l
    chunksize = 20
elif device == 'kraken':
    data_dir = '/scratch/cfd/ajuria/Plasma/plasmanet/datasets/IRSPR/fourier_singlemode_10/' + casename_l
    chunksize = 5

fig_dir = data_dir + 'figures/'
create_dir(data_dir)
create_dir(fig_dir)

def params(nits, unfiltered_mode):
    """ Parameters to give to compute function for imap """
    for i in range(nits):
        random_array = np.random.random((poisson.nmax, poisson.mmax))
        rhs_coefs = rhs0 * (2 * random_array - 1)
        rhs_coefs *= 0.1
        rhs_coefs[unfiltered_mode,unfiltered_mode] *= 10
        # yield rhs_coefs / (poisson.N**2 + poisson.M**2)
        yield rhs_coefs

def compute(args):
    """ Compute function for imap (multiprocessing) """
    rhs_coefs = args
    # interior rhs
    tmp_potential = poisson.pot_series(rhs_coefs)
    tmp_rhs = poisson.sum_series(rhs_coefs)

    return tmp_potential, tmp_rhs


if __name__ == '__main__':

    # Print header of dataset
    print(f'Device : {device:s} - npts = {npts:d} - nits = {nits:d} - nmax_fourier = {nmax_fourier:d}')
    print(f'Directory : {data_dir:s} - n_procs = {n_procs:d} - chunksize = {chunksize:d}')

    potential_list = np.zeros((nits, npts, npts))
    physical_rhs_list = np.zeros((nits, npts, npts))

    time_start = time.time()

    with get_context('spawn').Pool(processes=n_procs) as p:
        results_train = list(tqdm(p.imap(compute, params(nits, unfiltered_mode), chunksize=chunksize), total=nits))

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