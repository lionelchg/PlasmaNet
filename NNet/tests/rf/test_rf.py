########################################################################################################################
#                                                                                                                      #
#                                           Test the divergence operator                                               #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 28.02.2020                                #
#                                                                                                                      #
########################################################################################################################


import torch
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib import ticker

import PlasmaNet.nnet.model as module_arch
from PlasmaNet.nnet.parse_config import ConfigParser


import os
import argparse
import collections
import yaml

import pdb

from pathlib import Path



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

            in_res = cfg_dict['arch']['args']['input_res'] 

            # Creation of config object
            config = ConfigParser(cfg_dict, False)

            logger = config.get_logger('train')

            # Build model architecture, then print to console
            model = config.init_obj('arch', module_arch)
            #logger.info(model)

            def weights_init(m):
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.ones_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            model.apply(weights_init)


            data = torch.zeros((1,1, in_res, in_res))
            data[:, :, in_res//2, in_res//2] = 1


            # Evaluate the network and follow metrics
            #with torch.no_grad():
            output = model(data)

            output = torch.where(output > 0, 1.0, 0.0)
      

            x = [in_res//2 - model.rf_global//2, in_res//2 - model.rf_global//2, in_res//2 + model.rf_global//2, in_res//2 + model.rf_global//2, in_res//2 - model.rf_global//2]
            y = [in_res//2 - model.rf_global//2, in_res//2 + model.rf_global//2, in_res//2 + model.rf_global//2, in_res//2 - model.rf_global//2, in_res//2 - model.rf_global//2]

            fig, axarr = plt.subplots(2, 1, figsize=(6, 12))
            ax1, ax2 = axarr.ravel()
            p1 = ax1.contourf(data[0, 0, :, :], 100, cmap = 'Blues')
            cbar1 = fig.colorbar(p1, label='Input Field', ax=ax1)
            p2 = ax2.contourf(output[0, 0, :, :], 100,  cmap = 'Blues')
            ax2.plot(x,y, linestyle = 'dashed', color = 'red', linewidth=5)

            ax1.set_ylim(in_res//2 - model.rf_global, in_res//2 + model.rf_global)
            ax1.set_xlim(in_res//2 - model.rf_global, in_res//2 + model.rf_global)
            ax2.set_ylim(in_res//2 - model.rf_global, in_res//2 + model.rf_global)
            ax2.set_xlim(in_res//2 - model.rf_global, in_res//2 + model.rf_global)
            cbar2 = fig.colorbar(p2, label='Output field', ax=ax2,)
            plt.tight_layout()
            plt.savefig('test_{}_rf_{}_k_{}.png'.format(network, model.rf_global, model.kernel_sizes[0]))
            plt.close()


            # Correct Calculating Mehtod
            img_np = np.ones((1,1,in_res, in_res))
            img_ = torch.from_numpy(img_np).clone().detach().requires_grad_(True)
            out_cnn=model(img_.clone())
            out_shape=out_cnn.size()
            ndims=len(out_cnn.size())
            grad=torch.zeros(out_cnn.size())
            l_tmp=[]
            for i in range(ndims):
                if i==0 or i ==1:#batch or channel
                    l_tmp.append(0)
                else:
                    l_tmp.append(out_shape[i]//2)
    
            grad[tuple(l_tmp)]=0.1
            out_cnn.backward(gradient=grad)
            grad_np=img_.grad[0,0].data.numpy()
            idx_nonzeros=np.where(grad_np!=0)
            RF=[np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros]

            print('Originally Calculated RF: {}'.format(model.rf_global))
            print('Inversed Procedure RF: {}'.format(int(torch.sum(output)**0.5)))  
            print('Antonio style (correct): {}'.format(RF[0]))


    #assert torch.sum(output) == model.rf_global * model.rf_global


if __name__ == '__main__':

    test_rf_2d()

