########################################################################################################################
#                                                                                                                      #
#                                Simple network  with multiple scale for testing purposes                              #
#                                                                                                                      #
#                                          Ekhi Ajuria, CERFACS, 23.10.2020                                            #
#                                                                                                                      #
########################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel

class _Convscale(nn.Module):
    """
    Basic Convolution block.
    Input:
    - data_channels: number of input images
    - filters: number of filters in the only layer of the network.

    The network will only have two conv2D layers with an intermediate activation function at each scale.
    Each layer will have a kernel of size 3x3 and no stride. Padding is only done with replication padding.
    The order is: Pad Conv ReLu Pad Conv.
    """
    def __init__(self, data_channels, filters):
        super(_Convscale, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(data_channels, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=0)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class SimpleScaleNet(BaseModel):
    """
    Define the network. The inputs needed are:
    - data_channels (int): The number of data (input) channels
    - filters (int): number of filters of each layer
    - scales (int): number of scales
    - concat (bool): boolean to check if the output of smaller scales is injected for high resolution scales.
    - add_scales (bool): boolean to check if scales are added or not (old behavior)
    The network will only have two conv2D layers with an intermediate activation function at each scale.
    The order is: Pad Conv ReLu Pad Conv
    Depending on the number of scales added, the network will consequently interpolate the domain as input.
    All the outputs will be added at the end.
    """
    def __init__(self, data_channels, filters, scales, concat, add_scales):
        super(SimpleScaleNet, self).__init__()

        self.scales = scales
        self.concat = concat
        self.add_scales = add_scales
        self.scales_module = nn.ModuleList()
        self.output_list = []
        
        # Lowest resolution scale
        self.scales_module.append(_Convscale(data_channels, filters))

        for i in range(self.scales-1):
            if self.concat:
                self.scales_module.append(_Convscale(data_channels+1, filters))
            else:
                self.scales_module.append(_Convscale(data_channels, filters))


    def forward(self, x, epoch):
        """
        Inputs:
        - x (torch.tensor): Tensor containing input fields (bsz, channels, h, w)
        - epoch (int): Value corresponding to the actual epoch 

        y[0] corresponds to the definitive result!

        y[1] corresponds to the lowest resolution output (interpolated to the original size)
        y[scales]  corresponds to the highest resolution output
        """

        y = torch.zeros_like(x[:,0].unsqueeze(1).expand(-1, self.scales+1, -1, -1))
        for i, scale_layer in enumerate(self.scales_module):

            size_int = [int(j * (1/(2**((self.scales-i))))) for j in list(x.size()[2:])]
            if self.concat and i > 0:
                x_out = scale_layer(torch.cat((F.interpolate(x, size_int, mode='bilinear', align_corners=False),
                                                F.interpolate(x_out, size_int, mode='bilinear', align_corners=False)), dim=1)) 
            else:
                x_out = scale_layer(F.interpolate(x, size_int, mode='bilinear', align_corners=False)) 

            y[:,i+1] = F.interpolate(x_out, x.size()[2:], mode='bilinear', align_corners=False)[:,0] 

        if self.add_scales:
            y[:,0] = y.sum(1)
        else:
            y[:,0] = y[:,-1]  

        return y
