########################################################################################################################
#                                                                                                                      #
#                                         Drift-diffusion fluid plasma solver                                          #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 09.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Standard libraries
import numpy as np
import yaml
import scipy.constants as co

# Solver library
from plasma import StreamerMorrowDL

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


def main(cfg_streamer, cfg_dl):
    """ Main function containing initialisation, temporal loop and outputs. 
            Takes a configuration dict as input. """

    # Load the network
    logger = cfg_dl.get_logger('test')

    # Setup data_loader instances
    data_loader = cfg_dl.init_obj('data_loader', module_data)

    # Build model architecture
    model = cfg_dl.init_obj('arch', module_arch)

    # Get function handles of loss and metrics
    loss_fn = cfg_dl.init_obj('loss', module_loss)
    metric_fns = [getattr(module_metric, metric) for metric in cfg_dl['metrics']]

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

    sim = StreamerMorrowDL(cfg_streamer)

    # Print information
    sim.print_init()

    # Iterations
    for it in range(1, sim.nit + 1):
        sim.dtsum += sim.dt

        # Solve poisson equation from charge distribution
        sim.solve_poisson_dl(model)
        # sim.solve_poisson()

        # Update of the residual to zero
        sim.resnd[:] = 0

        # Compute the chemistry source terms with or without photo
        sim.compute_chemistry(it)

        # Compute transport terms
        sim.compute_residuals()

        # Apply residuals
        sim.update_res()

        # Post processing of macro values
        sim.global_prop(it)

        # General post processing
        sim.postproc(it)

    sim.plot_global()
    np.save(sim.data_dir + 'globals', sim.gstreamer)


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PlasmaNet')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to checkpoint to resume (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # Custom CLI options to modify configuration from default values given in yaml file
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-ds', '--dataset'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['-n', '--name'], type=str, target='name')
    ]
    cfg_dl = ConfigParser.from_args(args, options)

    with open('dh_streamer.yml', 'r') as yaml_stream:
        cfg_streamer = yaml.safe_load(yaml_stream)

    main(cfg_streamer, cfg_dl)
