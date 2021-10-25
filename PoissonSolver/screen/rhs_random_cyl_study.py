########################################################################################################################
#                                                                                                                      #
#                  2D photo datasets using random generation of rhs points for cylindrical geometries                  #
#                     This dataset plots lambda values varying from 0 to lambda_max (on config file)                   #
#                                 Ekhi Ajuria and Lionel Cheng, CERFACS, 20.09.2021                                    #
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

from PlasmaNet.poissonscreensolver.photo_ls import PhotoLinSystem
from PlasmaNet.common.utils import create_dir
from PlasmaNet.common.operators_numpy import grad, lapl

args = argparse.ArgumentParser(description='RHS random dataset')
args.add_argument('-c', '--cfg', type=str, default=None,
                help='Config filename')
args.add_argument('-nn', '--nnodes', default=None, type=int,
                    help='Number of nodes in x and y directions')
args.add_argument('--case', type=str, default=None, help='Case name')

# Specific arguments
args.add_argument('-nr', '--n_res_factor', default=24, type=int,
                    help='grid of npts/nres on which the random set is taken')
args = args.parse_args()

with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

device = cfg['device']
nits = cfg['n_entries']
n_procs = cfg['n_procs']

# Overwrite the resolution if in CLI
if args.nnodes is not None:
    cfg['photo']['nnx'] = args.nnodes
    cfg['photo']['nny'] = args.nnodes

photo = PhotoLinSystem(cfg['photo'])

xmin, xmax, nnx = photo.xmin, photo.xmax, photo.nnx
ymin, ymax, nny = photo.ymin, photo.ymax, photo.nny
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
l_max = cfg['photo']['lambda_max']

def params(nits, l_max):
    """ Parameters to give to compute function for imap """
    for i in range(nits):
        np.random.seed(seed=17)
        z_lower = 2 * np.random.random((nny_lower, nnx_lower)) - 1
        f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')
        lamb = i*l_max/nits
        yield [f(x, y), lamb]


def compute(args):
    """ Compute function for imap (multiprocessing) """
    ioniz_rate = ni0 * args[0]
    lamb_field = args[1] * np.ones_like(ioniz_rate)
    data = np.stack([ioniz_rate, lamb_field], axis=0)
    Sph = photo.solve(data, bcs)

    return Sph, data

if __name__ == '__main__':
    # Parameters for the rhs and plotting
    plot = True
    plot_period = 1
    freq_period = 1

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

    Sph_list = np.zeros((nits, nny, nnx))
    ioniz_rate_list = np.zeros((nits, nny, nnx))

    time_start = time.time()

    with get_context('spawn').Pool(processes=n_procs) as p:
        results_train = list(tqdm(p.imap(compute, params(nits, l_max), chunksize=chunksize), total=nits))

    for i, (photo_sph, data) in enumerate(tqdm(results_train)):
        Sph_list[i, :, :] = photo_sph
        ioniz_rate_list[i, :, :] = data[0]/(photo.dx * photo.dy)
        if i % plot_period == 0:
            if i ==0:
                lapl_field = -lapl(photo_sph, photo.dx, photo.dy, photo.nnx, photo.nny, r=photo.R_nodes)
                photo.plot_2D_variable(fig_dir + f'input_lapl_{i:05d}', photo_sph, lapl_field, data[1, 0, 0], geom='xr', axis='on')
            photo.plot_2D_variable(fig_dir + f'input_{i:05d}', photo_sph, data[0]/(photo.dx * photo.dy), data[1, 0, 0], geom='xr', axis='on')

    np.save(data_dir + 'Sph.npy', Sph_list)
    np.save(data_dir + 'ioniz_rate.npy', ioniz_rate_list)
    time_stop = time.time()
    print('Elapsed time (s) : %.2f' % (time_stop - time_start))
