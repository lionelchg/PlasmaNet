########################################################################################################################
#                                                                                                                      #
#                                   ScalesNet class - base for both UNet and MSNet                                     #
#                                                                                                                      #
#                                  Ekhi Ajuria & Lionel Cheng, CERFACS, 05.05.2021                                     #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import torch
from ..base import BaseModel


class ScalesNet(BaseModel):
    def __init__(self, scales: dict, kernel_sizes):
        super(ScalesNet, self).__init__()
        # Main parameters
        self.scales = scales

        # Handle non-square kernels
        if isinstance(kernel_sizes, int):
            self.kernel_sizes = [tuple([kernel_sizes, kernel_sizes])] * len(scales)
        elif isinstance(kernel_sizes, list):
            if isinstance(kernel_sizes[0], list):
                self.kernel_sizes = [tuple(kernel_sizes[0])] * len(scales)
            else:
                # Convert the list of integers to a list of tuples
                self.kernel_sizes = [tuple([ks, ks]) for ks in self.kernel_sizes]

        # Global parameters of the network
        self.n_scales = len(scales)
        self.max_scale = self.n_scales - 1

        # Create width of the network and treat list of list cases
        # happening for UNet
        self.depths = []
        self.params = []
        for iscale, scale in enumerate(scales.values()):
            tmp_list = list()
            tmp_params = 0
            if isinstance(scale[0], list):
                for elem in scale:
                    tmp_list += elem
                    tmp_params += sum([elem[i] * elem[i + 1] * (self.kernel_sizes[iscale][0] * self.kernel_sizes[iscale][1]) + elem[i + 1] for 
                        i in range(len(elem) - 1)])
                self.depths.append(len(tmp_list) - 2)
            else:
                tmp_list = scale
                tmp_params += sum([scale[i] * scale[i + 1] * (self.kernel_sizes[iscale][0] * self.kernel_sizes[iscale][1]) + scale[i + 1] for 
                        i in range(len(scale) - 1)])
                self.depths.append(len(tmp_list) - 1)
            self.params.append(tmp_params)

        # Compute the receptive field of the network in x and y directions
        self.receptive_field_x = list()
        self.receptive_field_y = list()
        for s in range(self.n_scales):
            self.receptive_field_x.append(self.depths[s] * (self.kernel_sizes[s][1] - 1) * 2**s)
            self.receptive_field_y.append(self.depths[s] * (self.kernel_sizes[s][0] - 1) * 2**s)
        # + 1 for initial point
        self.rf_global_x = 1 + sum(self.receptive_field_x)
        self.rf_global_y = 1 + sum(self.receptive_field_y)

    def global_prop(self):
        """ Global properties of the network """
        nscales_str = f'Number of branches of the network: {self.n_scales:d}'
        prop_str = 'Global properties of each scale: \n'
        prop_str += (f'{" ":10} |{"RF_x":^10}|{"RF_y":^10}|{"Depth":^10}|{"nparams":^10}|'
                f'{"k-size-x":^10}|{"k-size-y":^10}|')
        for s in range(self.n_scales):
            prop_str += f'\nBranch {s:d}   |{self.receptive_field_x[s]:10d}|{self.receptive_field_y[s]:10d}|{self.depths[s]:10d}|{self.params[s]:10d}|{self.kernel_sizes[s][1]:10d}|{self.kernel_sizes[s][0]:10d}|'
        prop_str += f'\nTotal      |{self.rf_global_x:10d}|{self.rf_global_y:10d}|{sum(self.depths):10d}|{sum(self.params):10d}|'
        return '\n'.join([nscales_str, prop_str])

    def __str__(self):
        """ Print model with number of trainable parameters as well as
        depth and receptive field """
        base_str = super().__str__()
        global_prop_str = self.global_prop()
        return '\n'.join([base_str, global_prop_str])
