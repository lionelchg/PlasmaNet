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
        if isinstance(kernel_sizes, int):
            self.kernel_sizes = [kernel_sizes] * len(scales)

        # Global parameters of the network
        self.n_scales = len(scales)
        self.depth = self.n_scales - 1

        # Create width of the network and treat list of list cases
        # happening for UNet
        self.width = []
        for scale in scales.values():
            tmp_list = list()
            if isinstance(scale[0], list):
                for elem in scale:
                    tmp_list += elem
                self.width.append(len(tmp_list) - 2)
            else:
                tmp_list = scale
                self.width.append(len(tmp_list) - 1)

        # Compute the receptive field of the network
        self.receptive_field = list()
        for d in range(self.n_scales):
            self.receptive_field.append(self.width[d] * (self.kernel_sizes[d] - 1) * 2**d)
        # + 1 for initial point
        self.rf_global = 1 + sum(self.receptive_field)

    def __str__(self):
        """ Print model with number of trainable parameters as well as
        depth and receptive field """
        base_str = super().__str__()
        depth_str = f'Depth of the network: {self.depth:d}'
        rf_str = 'Params of each scale: \n   rf - width - kernel_size'
        for d in range(self.n_scales):
            rf_str += f'\n d{d:d}: {self.receptive_field[d]:d} - {self.width[d]:d} - {self.kernel_sizes[d]:d}'
        rf_str += f'\nTotal: {self.rf_global:d} - {sum(self.width):d}'
        return '\n'.join([base_str, depth_str, rf_str])