########################################################################################################################
#                                                                                                                      #
#                               Convective vortex for validation of Euler integration                                  #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
import yaml

from plasma import PlasmaEulerDL

# For network
import torch
import argparse
import collections

import PlasmaNet.data.data_loaders as module_data
import PlasmaNet.model.loss as module_loss
import PlasmaNet.model.metric as module_metric
from PlasmaNet.parse_config import ConfigParser
from PlasmaNet.trainer.trainer import plot_batch
import PlasmaNet.model as module_arch

# @profile
def run(config):
    """ Main function containing initialization, temporal loop and outputs. Takes a config dict as input. """
    cfg_dl = ConfigParser(config['network'])

    # Load the network
    logger = cfg_dl.get_logger('test')

    # Setup data_loader instances
    data_loader = cfg_dl.init_obj('data_loader', module_data)

    # Build model architecture
    model = cfg_dl.init_obj('arch', module_arch)

    logger.info('Loading checkpoint: {} ...'.format(cfg_dl['resume']))
    checkpoint = torch.load(cfg_dl['resume'])
    state_dict = checkpoint['state_dict']
    if cfg_dl['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()    
    
    sim = PlasmaEulerDL(config['plasma'], config['network'])
    # Print header to sum up the parameters
    if sim.verbose:
        sim.print_init()

    # Iterations
    for it in range(1, sim.nit + 1):
        sim.it = it
        sim.dtsum += sim.dt
        sim.time[it - 1] = sim.dtsum
        
        # Update of the residual to zero
        sim.res[:], sim.res_c[:] = 0, 0

        # Solve poisson equation
        sim.solve_poisson_dl(model)

        # Compute euler fluxes (without pressure)
        sim.compute_flux_cold()

        # Compute residuals in cell-vertex method
        sim.compute_res()

        # Compute residuals from electro-magnetic terms
        sim.compute_EM_source()

        # boundary conditions
        sim.impose_bc_euler()
        
        # Apply residual
        sim.update_res()

        # Post processing
        sim.postproc(it)

        # Retrieve center variables 
        sim.temporal_variables(it)

    # Plot temporals
    sim.post_temporal()

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PlasmaNet')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    run(config)