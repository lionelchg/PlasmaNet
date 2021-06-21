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
args.add_argument('-nmin', '--nmin', default=1, type=int,
                    help='Minimum mode number')
args.add_argument('-nmax', '--nmax', default=5, type=int,
                    help='Maximum mode number')
args.add_argument('-dp', '--decrease_power', default=2, type=int,
                    help='Decreasing power of the modes (polynomial)')
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

cfg['poisson']['nmin_fourier'] = args.nmin
cfg['poisson']['nmax_fourier'] = args.nmax

poisson = DatasetPoisson(cfg['poisson'])
zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)

# amplitude of the random modes
ni0 = 1e11
rhs0 = ni0 * co.e / co.epsilon_0


def params(nits):
    """ Parameters to give to compute function for imap """
    for i in range(nits):
        random_array = np.random.random((len(poisson.mrange), len(poisson.nrange)))
        rhs_coefs = rhs0 * (2 * random_array - 1)
        yield rhs_coefs / (poisson.N**args.decrease_power + poisson.M**args.decrease_power)


def compute(args):
    """ Compute function for imap (multiprocessing) """
    rhs_coefs = args
    # interior rhs
    tmp_potential = poisson.pot_series(rhs_coefs)
    tmp_rhs = poisson.sum_series(rhs_coefs)

    return tmp_potential, tmp_rhs


if __name__ == '__main__':
    # Parameters for the plotting
    plot = True
    plot_period = int(0.1 * nits)
    freq_period = int(0.01 * nits)

    # Directories declaration and creation if necessary
    if args.case is not None:
        casename = args.case + "/"
    else:
        casename = f'{poisson.nnx:d}x{poisson.nny:d}/fourier_{poisson.nmin:d}_{poisson.nmax:d}_{args.decrease_power:d}/'


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
    print(f'Device : {device:s} - nits = {nits:d} - nmax_fourier = {poisson.nmax:d}')
    print(f'Directory : {data_dir:s} - n_procs = {n_procs:d} - chunksize = {chunksize:d}')

    potential_list = np.zeros((nits, poisson.nny, poisson.nnx))
    physical_rhs_list = np.zeros((nits, poisson.nny, poisson.nnx))
    
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

    time_stop = time.time()
    np.save(data_dir + 'potential.npy', potential_list)
    np.save(data_dir + 'physical_rhs.npy', physical_rhs_list)
    print('Elapsed time (s) : %.2e' % (time_stop - time_start))
