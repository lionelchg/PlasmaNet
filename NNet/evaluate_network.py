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

import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


import PlasmaNet.nnet.data.data_loaders as module_data
import PlasmaNet.nnet.model.loss as module_loss
import PlasmaNet.nnet.model.metric as module_metric
import PlasmaNet.nnet.model as module_arch
from PlasmaNet.nnet.parse_config import ConfigParser
from PlasmaNet.nnet.trainer.trainer import plot_batch
from PlasmaNet.common.utils import create_dir


def main(config):

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

    return total_metrics/ n_samples, total_loss / n_samples


def plot_ticks(ax, labels_x, labels_y):
    """Useful functions to declare figure axes

    Args:
        ax (plt.ax): Axes on which ticks are plotted
        labels_x (list): List of strings with names
        labels_y (list): List of strings with names
    """
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_yticks(np.arange(len(labels_y)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels_x)
    ax.set_yticklabels(labels_y)
    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #        rotation_mode="anchor")

class PlotRes:
    """ Class that contains three plots performed for Resolution plots
    """
    def __init__(self, losses_tests, test_cases, spatial_resolutions, loss_titles, fig_dir, max_val, min_val):
        """ Plots the MSE of all test cases, the average of gaussian and random test cases
        and an the global mean.

        Args:
            losses_tests (torch.tensor): torch tensor containing all the losses to plot
            test_cases (list):  list of strings containing  the names of all the test cases evaluated
            spatial_resolutions (list): list of integers containing all the evaluated resolutions
            loss_titles (list): list of strings containing the name of the evaluated metrics
            fig_dir (string): directory on which the plots are saved
            max_val (float): max value of tensors for plotting
            min_val (float): min value of tensors for plotting
        """

        self.data = losses_tests
        self.cases = test_cases
        self.resolutions = spatial_resolutions
        self.losses = loss_titles
        self.dir = fig_dir
        self.max_values = max_val
        self.min_values = min_val
    
    def generate_four(self, fig):
        """ Returns 4 axes for the 4 subplots of the image

        Args:
            fig (plt.fig): Figure to be divided in 4 subplots
        """

        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[1, 0]) 
        ax2 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, 1])

        return [ax1, ax2, ax3, ax4]
    
    def define_log(self, log):
        """ Helpful method for plotting log_scale or not

        Args:
            log (bool): If true, Imshow in Log Scale
        """
        if log:
            norm_plot = LogNorm()
        else:
            norm_plot = None
        return norm_plot

    
    def plot_all(self, fig_size, log):

        fig = plt.figure(constrained_layout=True, figsize=fig_size)
        axes_all = self.generate_four(fig)

        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color = 'k', alpha = 1.)

        cmap_e = plt.get_cmap('Blues')
        cmap_e.set_bad(color = 'k', alpha = 1.)

        for i, ax in enumerate(axes_all):
            # Initial time step masked
            if i < 2:
                im = ax.imshow(self.data[:,:,i+1], vmin=self.min_values[i+1], vmax=self.max_values[i+1], norm=self.define_log(log), cmap=cmap)
            else:
                im = ax.imshow(self.data[:,:,i+1], vmin=self.min_values[i+1], vmax=self.max_values[i+1], norm=self.define_log(log), cmap=cmap_e)

            # Create colorbars 
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            cbar1 = ax.figure.colorbar(im, ax=ax, cax=cax)

            # Only put spatial resolution ticks on the first column
            if i==2 or i==3:
                ax.set_xlabel('Spatial Resolution')
            if i==0 or i==2: 
                plot_ticks(ax, self.resolutions, self.cases)
            else:
                plot_ticks(ax, self.resolutions, [])
            ax.set_title('{}'.format(self.losses[i]))

        # Save figure
        fig.savefig(self.dir + 'MSE_all_losses.png', bbox_inches='tight')
        plt.close()

    def plot_categories(self, test_cases_reduced, fig_size, log):

        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color = 'k', alpha = 1.)

        cmap_e = plt.get_cmap('Blues')
        cmap_e.set_bad(color = 'k', alpha = 1.)

        # Initialize grids, labels and colormaps
        self.test_cases_reduced = ['Gaussians', 'Random']
        fig = plt.figure(constrained_layout=True, figsize=fig_size)

        axes_all = self.generate_four(fig)

        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color = 'k', alpha = 1.)

        # Create and store in new tensor
        losses_section = torch.zeros((2, len(self.resolutions), 5))
        losses_section[0,:,:] = torch.mean(self.data[:3,:], 0)
        losses_section[1,:,:] = torch.mean(self.data[3:,:], 0)
        self.losses_section = losses_section

        for i, ax in enumerate(axes_all):
            # Initial time step masked
            if i<2:
                im = ax.imshow(losses_section[:,:,i+1], vmin=self.min_values[i+1], vmax=self.max_values[i+1], norm=self.define_log(log), cmap=cmap)
            else:
                im = ax.imshow(losses_section[:,:,i+1], vmin=self.min_values[i+1], vmax=self.max_values[i+1], norm=self.define_log(log), cmap=cmap_e)

            # Create colorbars and orientate ticks
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            cbar1 = ax.figure.colorbar(im, ax=ax, cax=cax)

            plot_ticks(ax, self.resolutions, self.test_cases_reduced)

            # Only put spatial resolution ticks on the first column
            if i ==2 or i == 3:
                ax.set_xlabel('Spatial Resolution')

            if i==1 or i==3:
                plot_ticks(ax, self.resolutions, [])
            else:
                plot_ticks(ax, self.resolutions, self.test_cases_reduced)

            ax.set_title('{}'.format(self.losses[i]))

        fig.savefig(self.dir + 'MSE_category_losses.png', bbox_inches='tight')
        plt.close()

    def plot_overall_mean(self, fig_size, log):
        # Single Column Plot Mean of Gaussians and Randoms
        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color = 'k', alpha = 1.)

        cmap_e = plt.get_cmap('Blues')
        cmap_e.set_bad(color = 'k', alpha = 1.)

        # Initialize Figure and colormaps
        fig = plt.figure(constrained_layout=True, figsize=fig_size)

        axes_all = self.generate_four(fig)

        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color = 'k', alpha = 1.)

        # Loop over axes
        for i, ax in enumerate(axes_all):
            # Initial time step masked
            if i<2:
                im = ax.imshow(torch.mean(self.losses_section[:,:,i+1], 1).unsqueeze(1), vmin=self.min_values[i+1], vmax=self.max_values[i+1], norm=self.define_log(log), cmap=cmap)
            else:
                im = ax.imshow(torch.mean(self.losses_section[:,:,i+1], 1).unsqueeze(1), vmin=self.min_values[i+1], vmax=self.max_values[i+1], norm=self.define_log(log), cmap=cmap_e)

            # Create colorbars and orientate ticks
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="20%", pad=0.05)
            cbar1 = ax.figure.colorbar(im, ax=ax, cax=cax)
            plot_ticks(ax, ['Mean'], self.test_cases_reduced)

            # Ax format
            #ax.set_xlabel('Spatial Resolution')
            ax.set_title('{}'.format(self.losses[i]))

        # Save figure and 3 arrays
        fig.savefig(self.dir + 'MSE_mean.png', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PlasmaNet.nnet')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args = args.parse_args()

    original_config = args.config 


    # Define test cases and resolutions on which networks will be evaluated
    test_cases = ['gaussian_centered',
                  'gaussian_offcentered',
                  'double_gaussian',
                  'random_4',
                  'random_6',
                  'random_8',
                  'random_10', 
                  'random_12',
                  'random_14']
    spatial_resolutions = [61, 81, 101, 121, 141, 201]
    loss_titles = [r'Loss_mean', r'Loss', r'$|| \phi_{NN} - \phi_{target} ||_{1}$', r'$|| \phi_{NN} - \phi_{true} ||_{\infty}$', 
                                          r'$|| \vb{E}_{NN} - \vb{E}_{target} ||_{1}$', r'$|| \vb{E}_{NN} - \vb{E}_{target} ||_{\infty}$']

    plot_loss_titles= [r'$|| \phi_{NN} - \phi_{target} ||_{1}$', r'$|| \phi_{NN} - \phi_{true} ||_{\infty}$', 
                       r'$|| \mathbf{E}_{NN} - \mathbf{E}_{target} ||_{1}$', r'$|| \mathbf{E}_{NN} - \mathbf{E}_{target} ||_{\infty}$']

    with open(original_config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Initialization of variables
    losses_tests = torch.zeros((len(spatial_resolutions),len(test_cases),len(loss_titles)-1))
    saving_path = os.path.join(config['trainer']['save_dir'], config['name'])
    fig_dir = os.path.join(saving_path, 'figures/')
    if not os.path.isdir(saving_path):
        create_dir(saving_path)
    if not os.path.isdir(fig_dir):
        create_dir(fig_dir)

    # Losses_npy file to load or create
    loss_file_name = os.path.join(saving_path, 'losses_save.npy')

    if os.path.exists(loss_file_name):
        losses_tests = torch.from_numpy(np.load(loss_file_name)) 
    else:
        # Loop over all test cases
        for i, test in enumerate(test_cases):
            for j, n_res in enumerate(spatial_resolutions): 
        
                # Load base config with yaml and modify nnx and data_dir
                with open(original_config, 'r') as yaml_stream:
                    config = yaml.safe_load(yaml_stream)
                base_data_dir = config['data_loader']['args']['data_dir']
                config['data_loader']['args']['data_dir'] = os.path.join(base_data_dir, test, '{}x{}'.format(n_res, n_res))
                config['globals']['train_nnx'] = config['globals']['nnx']
                config['globals']['nnx'], config['globals']['nny'] = n_res, n_res
                config = ConfigParser(config)
                # Perform simulations
                losses_tests[j, i, 1:], losses_tests[j, i, 0] = main(config)
                plt.close('all')
    
    #pdb.set_trace()
    max_val = torch.max(torch.max(losses_tests, dim=0)[0],dim=0)[0]
    min_val = torch.min(torch.min(losses_tests, dim=0)[0],dim=0)[0] 

    # Load saving dirs, and create them if necessary
    #saving_path = '/scratch/cfd/ajuria/Plasma/cases/20_new_plasmanet/flexinet_new/pproc/debug/'

    # Save the array of size (spatial_res, test_cases, loss_titles-1) and 3 figures 
    np.save(loss_file_name, losses_tests)

    # Transpose for plotting correctly, and create the plotting class
    losses_tests = torch.transpose(losses_tests, 0, 1)
    plots = PlotRes(losses_tests, test_cases, spatial_resolutions, plot_loss_titles, fig_dir, max_val, min_val)

    plots.plot_all((10, 10), False)
    plots.plot_categories(['Gaussians', 'Random'],(8, 3), False)
    plots.plot_overall_mean((5,4), False)



