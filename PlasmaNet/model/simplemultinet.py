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

class _ConvscaleFlexible(nn.Module):
    """
    Fleixble.
    Input:
    - data_channels: number of input images
    - filters (list): number of filters in the network's layers.

    The network will have conv2D layers with an intermediate activation function at each scale.
    Each layer will have a kernel of size 3x3 and no stride. Padding is only done with replication padding.
    The order is: Pad Conv ReLu Pad Conv.
    """
    def __init__(self, data_channels, filters):
        super(_ConvscaleFlexible, self).__init__()

        # Module list containing all the layers
        self.scale_filters = nn.ModuleList()

        # Initialize with input size.
        self.scale_filters.append(nn.ReplicationPad2d(1))
        self.scale_filters.append(nn.Conv2d(data_channels, filters[0], kernel_size=3, stride=1, padding=0))

        # Intermediate layers.
        for i in range(len(filters)-1):
            self.scale_filters.append(nn.ReLU())
            self.scale_filters.append(nn.ReplicationPad2d(1))
            self.scale_filters.append(nn.Conv2d(filters[i], filters[i+1], kernel_size=3, stride=1, padding=0))

        # Final layer, note that there is no ReLU at the end.
        self.scale_filters.append(nn.ReLU())
        self.scale_filters.append(nn.ReplicationPad2d(1))
        self.scale_filters.append(nn.Conv2d(filters[-1], 1, kernel_size=3, stride=1, padding=0))

    def forward(self, x):
        for i in range(len(self.scale_filters)):
            x = self.scale_filters[i](x)
        return x


class SimpleScaleNet(BaseModel):
    """
    Define the network. The inputs needed are:
    - data_channels (int): The number of data (input) channels
    - filters (list of list)s: number of filters of each layer and scale, thus containing how many scales will be needed.
    The number of scales MUST correspond to the number of lists in the filters list.
    The first element of the list will corrrespond to the lowest resolution scale, whereas the last one will be the scale
    with the original size.
    - scales (int): number of scales
    - concat (bool): boolean to check if the output of smaller scales is injected for high resolution scales.
    - add_scales (bool): boolean to check if scales are added or not (old behavior)
    The network have as many conv2D layers with an intermediate activation function at each scale as indicated on the
    scales lists.
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

        assert len(filters) == scales, 'Scales and inputted filters do not match'

        # Lowest resolution scale
        self.scales_module.append(_ConvscaleFlexible(data_channels, filters[0]))

        for i in range(len(filters)-1):
            if self.concat:
                self.scales_module.append(_ConvscaleFlexible(data_channels+1, filters[i+1]))
            else:
                self.scales_module.append(_ConvscaleFlexible(data_channels, filters[i+1]))


    def forward(self, x):
        """
        Inputs:
        - x (torch.tensor): Tensor containing input fields (bsz, channels, h, w)

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