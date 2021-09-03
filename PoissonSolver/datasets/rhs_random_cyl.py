########################################################################################################################
#                                                                                                                      #
#                  2D Poisson datasets using random generation of rhs points for cylindrical geometries                #
#                       This dataset will be used for networks working on streamer simulations                         #
#                                          Lionel Cheng, CERFACS, 04.09.2021                                           #
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

args = argparse.ArgumentParser(description='RHS random dataset')
args.add_argument('-c', '--cfg', type=str, default=None,
                help='Config filename')
args.add_argument('-nn', '--nnodes', default=None, type=int,
                    help='Number of nodes in x and y directions')
args.add_argument('--case', type=str, default=None, help='Case name')

# Specific arguments
args.add_argument('-nr', '--n_res_factor', default=4, type=int,
                    help='grid of npts/nres on which the random set is taken')
args = args.parse_args()

with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

device = cfg['device']
nits = cfg['n_entries']
n_procs = cfg['n_procs']

# Overwrite the resolution if in CLI
if args.nnodes is not None:
    cfg['poisson']['nnx'] = args.nnodes
    cfg['poisson']['nny'] = args.nnodes

poisson = DatasetPoisson(cfg['poisson'])

xmin, xmax, nnx = poisson.xmin, poisson.xmax, poisson.nnx
ymin, ymax, nny = poisson.ymin, poisson.ymax, poisson.nny
x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)

# Factor to divide the grid by to generate the random grid
n_res_factor = args.n_res_factor

nnx_lower = int(nnx / n_res_factor)
nny_lower = int(nny / n_res_factor)
x_lower, y_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower)

# Amplitude of the RhS
ni0 = 1e16
bcs = {'left':zeros_y, 'right':zeros_y, 'top':zeros_x}

def params(nits):
    """ Parameters to give to compute function for imap """
    for i in range(nits):
        z_lower = 2 * np.random.random((nny_lower, nnx_lower)) - 1
        f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')
        yield f(x, y)


def compute(args):
    """ Compute function for imap (multiprocessing) """
    physical_rhs = ni0 * args * co.e / co.epsilon_0

    poisson.solve(physical_rhs, bcs)

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
        casename = f'{nnx:d}x{nny}/random_{n_res_factor:d}/'

    if device == 'mac':
        chunksize = 20
    elif device == 'kraken':
        chunksize = 5

    # Directories
    data_dir = cfg['output_dir'] + casename
    fig_dir = data_dir + 'figures/'
    create_dir(data_dir)
    create_dir(fig_dir)

    # Print header of dataset
    print(f'Casename : {casename:s}')
    print(f'Device : {device:s} - nits = {nits:d} - n_res_factor = {n_res_factor:d}')
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
            poisson.plot_2D(fig_dir + f'input_{i:05d}', axis='off')
        if i % freq_period == 0:
            poisson.physical_rhs = rhs
            poisson.compute_modes()

    poisson.plot_pmodes(fig_dir + 'average_modes')

    np.save(data_dir + 'potential.npy', potential_list)
    np.save(data_dir + 'physical_rhs.npy', physical_rhs_list)
    time_stop = time.time()
    print('Elapsed time (s) : %.2f' % (time_stop - time_start))
