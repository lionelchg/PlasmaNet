########################################################################################################################
#                                                                                                                      #
#                              2D Poisson datasets using random generation of rhs gaussian                             #
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
from tqdm import tqdm

from PlasmaNet.poissonsolver.poisson import DatasetPoisson
from PlasmaNet.common.utils import create_dir
from PlasmaNet.common.profiles import gaussian


args = argparse.ArgumentParser(description='RHS random dataset')
args.add_argument('-c', '--cfg', type=str, default=None,
                help='Config filename')
args.add_argument('-nn', '--nnodes', default=None, type=int,
                    help='Number of nodes in x and y directions')
args.add_argument('--case', type=str, default=None, help='Case name')
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

# Parameters for the rhs and plotting
ni0 = 1e11
rhs0 = ni0 * co.e / co.epsilon_0
ampl_min, ampl_max = 0.01, 1
sigma_min, sigma_max = 1e-3, 3e-3
x_middle_min, x_middle_max = 0.35e-2, 0.65e-2

bcs = {'left':zeros_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}

def params(nits):
    """ Parameters to give to compute function for imap """
    for i in range(nits):
        coefs = np.random.random(5)
        ampl = rhs0 * ((ampl_max - ampl_min) * coefs[0] + ampl_min)
        sigma_x = (sigma_max - sigma_min) * coefs[1] + sigma_min
        sigma_y = (sigma_max - sigma_min) * coefs[2] + sigma_min
        x_gauss = (x_middle_max - x_middle_min) * coefs[3] + x_middle_min
        y_gauss = (x_middle_max - x_middle_min) * coefs[4] + x_middle_min
        yield ampl, x_gauss, y_gauss, sigma_x, sigma_y


def compute(args):
    """ Compute function for imap (multiprocessing) """
    ampl, x_gauss, y_gauss, sigma_x, sigma_y = args
    physical_rhs = gaussian(poisson.X, poisson.Y, ampl, x_gauss, y_gauss, sigma_x, sigma_y)
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
        casename = f'{poisson.nnx:d}x{poisson.nny}/hills/'

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
    print(f'Device : {device:s} - nits = {nits:d}')
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
