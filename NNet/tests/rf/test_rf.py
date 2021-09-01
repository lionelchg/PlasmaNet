########################################################################################################################
#                                                                                                                      #
#                                        Test the Receptive field empirically                                          #
#                                                                                                                      #
#                                          Ekhi Ajuria, CERFACS, 28.05.2021 (mod. Victor Xing 01.09.21)                #
#                                                                                                                      #
########################################################################################################################


import torch
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib import ticker

from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

import PlasmaNet.nnet.model as module_arch
from PlasmaNet.nnet.parse_config import ConfigParser
from PlasmaNet.common.plot import plot_ax_scalar

import os
import argparse
import collections
import yaml

import pdb

from pathlib import Path


def plot_test(data, output, model, in_res, folder, network):

    # Create folder if does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create grids for plotting
    xx = np.arange(0, in_res)
    yy = np.arange(0, in_res)
    Xb, Yb = np.meshgrid(xx, yy)

    if 'center' in folder:
        x = [in_res//2 - model.rf_global//2, in_res//2 - model.rf_global//2, in_res//2 + model.rf_global//2, in_res//2 + model.rf_global//2, in_res//2 - model.rf_global//2]
        y = [in_res//2 - model.rf_global//2, in_res//2 + model.rf_global//2, in_res//2 + model.rf_global//2, in_res//2 - model.rf_global//2, in_res//2 - model.rf_global//2]
    else:
        x = [0, 0, model.rf_global//2, model.rf_global//2, 0]
        y = [0, model.rf_global//2, model.rf_global//2, 0, 0]

    # Image initialization
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(5.5,3))
    fig.subplots_adjust(wspace=0.3)

    # Plot input, output and expected box
    plot_ax_scalar(fig, ax1, Xb, Yb, data[0, 0, :, :].numpy(), r'Init')
    plot_ax_scalar(fig, ax2, Xb, Yb, output[0, 0, :, :].numpy(), r'RF')
    ax2.plot(x,y, linestyle = 'dashed', color = 'red', linewidth=5)

    # Cut the domain so that only the middle "interesting" part remains
    if 'center' in folder:
        ax1.set_ylim(in_res//2 - model.rf_global, in_res//2 + model.rf_global)
        ax1.set_xlim(in_res//2 - model.rf_global, in_res//2 + model.rf_global)
        ax2.set_ylim(in_res//2 - model.rf_global, in_res//2 + model.rf_global)
        ax2.set_xlim(in_res//2 - model.rf_global, in_res//2 + model.rf_global)
    else:
        ax1.set_ylim(0, model.rf_global)
        ax1.set_xlim(0, model.rf_global)
        ax2.set_ylim(0, model.rf_global)
        ax2.set_xlim(0, model.rf_global)


    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'test_{}_rf_{}_k_{}.png'.format(network, model.rf_global, model.kernel_sizes[0])), dpi=100)
    plt.close()


def inverse_method(model, cfg_dict, network, center, erf_thres=0.045, saving_folder= 'Images'):
    # Generate input tensor (all zeros except the middle point)
    in_res = cfg_dict['arch']['args']['input_res']
    data = torch.zeros((1,1, in_res, in_res))
    if center:
        data[:, :, in_res//2, in_res//2] = 1
        ctr = 'center'
    else:
        data[:, :, 0, 0] = 1
        ctr = 'BC'

    # Evaluate the network and follow metrics
    output = model(data).detach()

    # Plot
    plot_test(data, output, model, in_res, os.path.join(cfg_dict['name'],'figures' , 'Inverse', '{}'.format(ctr)), network)

    if center:
        thres = erf_thres * output[0, 0, in_res//2, in_res//2].numpy()
    else:
        thres = erf_thres * output[0, 0, 0, 0].numpy()

    erf_output = torch.where(output > thres, 1.0, 0.0)
    rf_output = torch.where(output > 0, 1.0, 0.0)

    return int(torch.sum(rf_output)**0.5), int(torch.sum(erf_output)**0.5)


def direct_method(model, cfg_dict, network, center, erf_thres=0.045, saving_folder= 'Images'):
    """ Direct RF calculating method issued from:
        https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/Receptive_Field.ipynb

    - model: network containing the wieghts initialized to 1  and biases to 0
    - cfg_dict: dictionary loaded from the cfg.yml file
    - network: string containing the name of the studied network
    - center: boolean to study the center or the BC case

    Effective receptive field threshold: 2 standard deviations from value at the signal
    location (cf Luo et al. NIPS 2016) """

    # Generate Input data
    in_res = cfg_dict['arch']['args']['input_res']
    img_ = torch.ones((1,1,in_res, in_res)).clone().detach().requires_grad_(True)

    # Evaluatre network
    out_cnn=model(img_.clone())

    # Generate gradient tensor which will be 0 everywhere except in one middle or BC point
    grad=torch.zeros(out_cnn.size())
    if center:
        grad[:, :, in_res//2, in_res//2] =0.1
        ctr = 'center'
    else:
        grad[:, :, 0, 0] =0.1
        ctr = 'BC'

    # Get the gradient only of the middle point!
    # Compute Receptive field
    out_cnn.backward(gradient=grad)
    grad_torch=img_.grad.detach()

    # Plot
    plot_test(grad, grad_torch, model, in_res, os.path.join(cfg_dict['name'],'figures' ,'Direct', '{}'.format(ctr)), network)

    if center:
        thres = erf_thres * grad_torch[0, 0, in_res//2, in_res//2].numpy()
    else:
        thres = erf_thres * grad_torch[0, 0, 0, 0].numpy()

    erf_grad_torch = torch.where(grad_torch > thres, 1.0, 0.0)
    rf_grad_torch = torch.where(grad_torch > 0, 1.0, 0.0)

    return int(torch.sum(rf_grad_torch)**0.5), int(torch.sum(erf_grad_torch)**0.5)


def test_rf_2d():

    torch.set_default_dtype(torch.float64)

    with open('cfg.yml', 'r') as yaml_stream:
        cfg_dict = yaml.safe_load(yaml_stream)


    for file in cfg_dict['files']:
        print('')
        print('-----------------------------------------------------------')
        print('')
        print('Entering file: {}'.format(file))
        print('')
        for network in cfg_dict['networks']:

            print('')
            print('Load Network: {}'.format(network))
            print('')

            # Model initialization!

            cfg_dict['arch']['db_file'] = file
            cfg_dict['arch']['name'] = network

            # Architecture parsing in database
            if 'db_file' in cfg_dict['arch']:
                with open(Path(os.getenv('ARCHS_DIR')) / cfg_dict['arch']['db_file']) as yaml_stream:
                    archs = yaml.safe_load(yaml_stream)

                if network in archs:
                    tmp_cfg_arch = archs[cfg_dict['arch']['name']]
                else:
                    print('Network {} does not exist on file {} ===> Skipping'.format(network, file))
                    break

                if 'args' in cfg_dict['arch']:
                    tmp_cfg_arch['args'] = {**cfg_dict['arch']['args'], **tmp_cfg_arch['args']}
                cfg_dict['arch'] = tmp_cfg_arch

            # Creation of config object
            config = ConfigParser(cfg_dict, False)

            # Build model architecture and initialize its weights with;
            # 0 in biases and 1 in weights
            model = config.init_obj('arch', module_arch)

            def weights_init(m):
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.constant_(m.weight, 0.1)
                    #torch.nn.init.ones_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            model.apply(weights_init)

            # Compute RF with direct and inverse methods for the center points
            dir_RF, dir_ERF = direct_method(model, cfg_dict, network, True)
            inv_RF, inv_ERF = inverse_method(model, cfg_dict, network, True)

            print("===================================")
            print('Originally Calculated RF: {}'.format(model.rf_global))
            print('Direct Method RF: {}'.format(dir_RF))
            print('Direct Method effective RF: {}'.format(dir_ERF))
            print('Inverse Procedure RF: {}'.format(inv_RF))
            print('Inverse Procedure effective RF: {}'.format(inv_ERF))
            print("===================================")

            # Compute RF with direct and inverse methods for the BC points
            dir_RF, dir_ERF = direct_method(model, cfg_dict, network, False)
            inv_RF, inv_ERF = inverse_method(model, cfg_dict, network, False)

            print('Originally Calculated RF: {}'.format(model.rf_global//2))
            print('Direct Method RF: {}'.format(dir_RF))
            print('Direct Method effective RF: {}'.format(dir_ERF))
            print('Inverse Procedure RF: {}'.format(inv_RF))
            print('Inverse Procedure effective RF: {}'.format(inv_ERF))
            print("===================================")

    #assert torch.sum(output) == model.rf_global * model.rf_global


if __name__ == '__main__':

    test_rf_2d()

