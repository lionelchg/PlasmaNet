########################################################################################################################
#                                                                                                                      #
#                                              PlasmaNet.nnet: evaluate model                                               #
#                                                                                                                      #
#                         Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria, CERFACS, 10.03.2020                         #
#                                                                                                                      #
########################################################################################################################

import argparse
import collections
import yaml
import pdb

import torch
import numpy as np
from tqdm import tqdm
import os

from pathlib import Path
import matplotlib.pyplot as plt

import PlasmaNet.nnet.data.data_loaders as module_data
import PlasmaNet.nnet.model.loss as module_loss
import PlasmaNet.nnet.model.metric as module_metric
import PlasmaNet.nnet.model as module_arch
from PlasmaNet.nnet.parse_config import ConfigParser
from PlasmaNet.nnet.trainer.trainer import plot_batch, plot_batch_Efield
from PlasmaNet.common.utils import create_dir

from plot_eval import PlotRes

def evaluate(config):

    logger = config.get_logger('test')

    # Setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # Build model architecture
    model = config.init_obj('arch', module_arch)

    # Load from directory, resume dir does not need to contain the full path to model_best.pth
    dir_list = os.listdir(config['resume'])
    logger.info('Loading checkpoint: {} ...'.format(os.path.join(config['resume'], dir_list[-1], "model_best.pth")))
    checkpoint = torch.load(os.path.join(config['resume'], dir_list[-1], "model_best.pth"))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Get function handles of loss and metrics
    loss_fn = config.init_obj('loss', module_loss)
    metric_fns = [getattr(module_metric, metric) for metric in config['metrics']]

    # Prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)

    # Output configuration
    out_dir = config.fig_dir

    with torch.no_grad():
        for i, (data, target, data_norm, target_norm) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            data_norm, target_norm = data_norm.to(device), target_norm.to(device)

            output = model(data)
            output = (config['globals']['train_nnx']**2 / config['globals']['nnx']**2) * output

            #
            # save sample images, or do something with output here
            #
            fig = plot_batch(output, target, data, 0, i, config)
            fig.savefig(out_dir / 'batch_{:05d}.png'.format(i), dpi=150, bbox_inches='tight')
            fig = plot_batch_Efield(output, target, data, 0, i, config)
            fig.savefig(out_dir / 'batch_Efield_{:05d}.png'.format(i), dpi=150, bbox_inches='tight')
            plt.close()

            # Computing loss, metrics on test set
            if loss_fn.require_input_data():
                loss = loss_fn(output, target, data=data, target_norm=target_norm, data_norm=data_norm)
            else:
                loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(metric_fns):
                total_metrics[j] += metric(output, target, config) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        metric.__name__: total_metrics[i].item() / n_samples for i, metric in enumerate(metric_fns)
    })
    logger.info(log)

    return total_metrics / n_samples


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PlasmaNet.nnet')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args = args.parse_args()

    cfg_fn = args.config 

    with open(cfg_fn, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Define test cases and resolutions on which networks will be evaluated
    datasets_type = config['datasets']['type']
    datasets_res = config['datasets']['res']

    # Initialization of variables
    residuals = torch.zeros((len(datasets_res), len(datasets_type), 4))
    saving_path = Path(config['network']['name'])
    fig_dir = saving_path / 'figures/'
    saving_path.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Losses_npy file to load or create
    residuals_fn = os.path.join(saving_path, 'residuals.npy')

    if os.path.exists(residuals_fn):
        residuals = torch.from_numpy(np.load(residuals_fn))
    else:
        # Loop over all test cases
        for i, case in enumerate(datasets_type):
            for j, n_res in enumerate(datasets_res): 
        
                # Load base config with yaml and modify nnx and data_dir
                with open(cfg_fn, 'r') as yaml_stream:
                    config = yaml.safe_load(yaml_stream)['network']
                
                # Add the name of the suffix of the directories
                suffix_dir = f'{n_res:d}x{n_res:d}/{case}'
                config['name'] += suffix_dir
                config['data_loader']['args']['data_dir'] += suffix_dir

                # Set the resolution of the training and the target case
                config['globals']['train_nnx'] = config['globals']['nnx']
                config['globals']['nnx'], config['globals']['nny'] = n_res, n_res
                config = ConfigParser(config)

                # Perform inference
                residuals[j, i, :] = evaluate(config)
    
    max_val = torch.max(torch.max(residuals, dim=0)[0],dim=0)[0]
    min_val = torch.min(torch.min(residuals, dim=0)[0],dim=0)[0]

    # Save the array of size (spatial_res, datasets_type, residuals_names-1) and 3 figures 
    np.save(residuals_fn, residuals)

    # Transpose for plotting correctly, and create the plotting class
    residuals = torch.transpose(residuals, 0, 1)

    residuals_names = [r'$|| \phi_{NN} - \phi_{target} ||_{1}$', 
                        r'$|| \phi_{NN} - \phi_{target} ||_{\infty}$', 
                        r'$|| \mathbf{E}_{NN} - \mathbf{E}_{target} ||_{1}$', 
                        r'$|| \mathbf{E}_{NN} - \mathbf{E}_{target} ||_{\infty}$']

    plots = PlotRes(residuals, datasets_type, datasets_res, 
        residuals_names, fig_dir, max_val, min_val)

    plots.plot_all((10, 10), False)
    # plots.plot_categories(['Gaussians', 'Random'], (8, 3), False)
    # plots.plot_overall_mean((5, 4), False)

