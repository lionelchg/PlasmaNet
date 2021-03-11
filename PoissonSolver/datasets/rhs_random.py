########################################################################################################################
#                                                                                                                      #
#                              2D Poisson datasets using random generation of rhs points                               #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import time
from multiprocessing import get_context
import argparse
import yaml

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
from scipy import interpolate
from tqdm import tqdm

from PlasmaNet.poissonsolver.poisson import DatasetPoisson
from PlasmaNet.common.utils import create_dir


args = argparse.ArgumentParser(description='Rhs random dataset')
args.add_argument('-d', '--device', default=None, type=str,
                    help='device on which the dataset is run')
args.add_argument('-ni', '--nits', default=None, type=int,
                    help='number of entries in the dataset')
args.add_argument('-nr', '--n_res', default=None, type=int,
                    help='grid of npts/nres on which the random set is taken')
args.add_argument('-np', '--n_procs', default=None, type=int,
                    help='number of procs')
args.add_argument('--case', type=str, default=None,
                help='Case name')
args = args.parse_args()

device = args.device
nits, n_res, n_procs = args.nits, args.n_res, args.n_procs

with open('poisson_ls_xy.yml', 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
poisson = DatasetPoisson(cfg)

xmin, xmax, nnx = poisson.xmin, poisson.xmax, poisson.nnx
ymin, ymax, nny = poisson.ymin, poisson.ymax, poisson.nny
x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)

nnx_lower = int(nnx / n_res)
nny_lower = int(nny / n_res)
x_lower, y_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower)

ni0 = 1e11


def params(nits):
    """ Parameters to give to compute function for imap """
    for i in range(nits):
        z_lower = 2 * np.random.random((nny_lower, nnx_lower)) - 1
        f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')
        yield f(x, y)


def compute(args):
    """ Compute function for imap (multiprocessing) """
    physical_rhs = ni0 * args.reshape(-1) * co.e / co.epsilon_0

    poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)

    return poisson.potential, poisson.physical_rhs


if __name__ == '__main__':
    # Parameters for the rhs and plotting
    plot = True
    plot_period = int(0.1 * nits)
    freq_period = int(0.1 * nits)

    # Directories declaration and creation if necessary
    if args.case is not None:
        casename = args.case
    else:
        casename = f'{nnx:d}x{nny}/random_{n_res:d}/'
        
    if device == 'mac':
        data_dir = 'outputs/' + casename
        chunksize = 20
    elif device == 'kraken':
        data_dir = '/scratch/cfd/cheng/DL/datasets/' + casename
        chunksize = 5

    fig_dir = data_dir + 'figures/'
    create_dir(data_dir)
    create_dir(fig_dir)

    # Print header of dataset
    print(f'Casename : {casename:s}')
    print(f'Device : {device:s} - nits = {nits:d} - n_res = {n_res:d}')
    print(f'Directory : {data_dir:s} - n_procs = {n_procs:d} - chunksize = {chunksize:d}')

    potential_list = np.zeros((nits, nny, nnx))
    physical_rhs_list = np.zeros((nits, nny, nnx))

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

    np.save(data_dir + 'potential.npy', potential_list)
    np.save(data_dir + 'physical_rhs.npy', physical_rhs_list)
    time_stop = time.time()
    print('Elapsed time (s) : %.2f' % (time_stop - time_start))
