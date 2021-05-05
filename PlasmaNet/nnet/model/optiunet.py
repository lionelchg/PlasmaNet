########################################################################################################################
#                                                                                                                      #
#                                  2.0 Simple flexible network  combining Unet and Msc                                 #
#                                                                                                                      #
#                                           Ekhi Ajuria, CERFACS, 16.03.2021                                           #
#                                                                                                                      #
########################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel

class _ConvSimple(nn.Module):
    """
    Simple Conv Block for init and outputs.
    The network will have a conv2D layer and an optional ReLU activation function.
    Each layer will have a kernel of size 3x3 and no stride. Padding is done with zero padding
    (default behavior of nn.Conv2d).
    """
    def __init__(self, in_features, out_features, activation_fu, pad_method):
        """ Initialization of the simple convolutional layer
        The order is: Pad Conv (ReLu Pad Conv) * len(filters).

        :param in_features: number of input feature maps
        :type in_features: int
        :param out_features: int number of output feature maps
        :type out_features: int
        :param activation_fu: optional ReLU activation
        :type activation_fu: bool
        :param pad_method: padding mode, to choose between 'circular', 'zeros', 'reflect', 'replicate'
        :type filters: str
        """
        super(_ConvSimple, self).__init__()

        # Module list containing all the layers
        self.scale_filters = nn.ModuleList()

        # Conv and Activation
        self.scale_filters.append(nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1, padding_mode=pad_method))
        if activation_fu:
           self.scale_filters.append(nn.ReLU()) 


    def forward(self, x):
        for i in range(len(self.scale_filters)):
            x = self.scale_filters[i](x)
        return x

class _ConvscaleFlexible(nn.Module):
    """
    Flexible convolutional layer.
    The network will have conv2D layers with an intermediate activation function at each scale.
    Each layer will have a kernel of size 3x3 and no stride. Padding is done with zero padding
    (default behavior of nn.Conv2d).
    """
    def __init__(self, n_filters, filter_size, up_block, pad_method):
        """ Initialization of the convolutional layer
        The order is: Pad Conv (ReLu Pad Conv) * len(filters).

        :param n_filters: int for the number of filters of each layer (same per block)
        :type n_filters: int
        :param filter_size: int for the size of each filter (same per block)
        :type filter_size: int
        :param up_block: if the block is an up block, first conv has double size!
        :type up_block: bool
        :param pad_method: padding mode, to choose between 'circular', 'zeros', 'reflect', 'replicate'
        :type filters: str
        """
        super(_ConvscaleFlexible, self).__init__()

        # Module list containing all the layers
        self.scale_filters = nn.ModuleList()

        # Initialize without ReLU
        if up_block:
            self.scale_filters.append(nn.Conv2d(2*filter_size, filter_size, kernel_size=3, stride=1, padding=1, padding_mode=pad_method))
        else:
            self.scale_filters.append(nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=1, padding_mode=pad_method))

        # Intermediate layers.
        for i in range(n_filters):
            self.scale_filters.append(nn.ReLU())
            self.scale_filters.append(nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=1, padding_mode=pad_method))


    def forward(self, x):
        for i in range(len(self.scale_filters)):
            x = self.scale_filters[i](x)
        return x

class OptiUnet(BaseModel):
    """
    Define the network with *data_channels* number of inputs while *filters* hold the size of 
    all the convolutional layers applied. The first element of the list will corrrespond to the lowest resolution scale, 
    whereas the last one will be the scale with the original size.

    The network is supposed symmetric, so the conv blocks of a same scale are equal, and
    have as many conv2D layers with an intermediate activation function at each scale as indicated on the
    scales lists.
    The order is: Pad Conv ReLu Pad Conv
    Depending on the number of scales added, the network will consequently interpolate the domain as input.
    All the outputs will be added at the end.
    """
    def __init__(self, data_channels, n_filters, filter_size, scales, pad_method='zeros'):
        """ Initialization of OptiUNet with data_channels in in

        :param data_channels: The nunmber of data (input) channels
        :type data_channels: int
        :param n_filters: int for the number of filters of each layer (same per block)
        :type n_filters: int
        :param filter_size: int for the size of each filter (same per block)
        :type filter_size: int
        :param scales: number of scales
        :type scales: int
        :param pad_method: padding mode, to choose between 'circular', 'zeros', 'reflect', 'replicate'
        :type filters: str (optional, default = zeros)
        """
        super(OptiUnet, self).__init__()

        self.scales = scales
        self.scales_module = nn.ModuleList()
        self.pad_method = pad_method
        self.output_list = []

        # Initialize highest resolution scale
        self.scales_module.append(_ConvSimple(data_channels, filter_size, True, self.pad_method))

        # Loop to append scales from highest to lowest resolution
        # As the higher and lower resolution have two blocks, the loop goes up to scales +1 and not scales -1
        for i in range(scales+1):
            self.scales_module.append(_ConvscaleFlexible(n_filters, filter_size, False, self.pad_method))

        # Descending loop where scales are added increasing the resolution
        for i in range(scales - 1):
            self.scales_module.append(_ConvscaleFlexible(n_filters, filter_size, True, self.pad_method))

        # Final layer to return an output of size 1, when resolution is at its highest
        self.scales_module.append(_ConvSimple(filter_size, 1, True, self.pad_method))


    def forward(self, x):
        """
        Inputs:
        - x (torch.tensor): Tensor containing input fields (bsz, channels, h, w)
        """
        in_conv = self.scales_module[0]
        out_conv = self.scales_module[-1]

        # Loop descending on resolution (start with highest)
        for i in range(self.scales):

            # Identify scale and get the size to interpolate
            scale_layer = self.scales_module[i+1]
            size_int = [int(j / 2**i) for j in list(x.size()[2:])]

            # Highest resolution scale only takes input
            if i == 0:
                x_out = in_conv(x)
                x_out = scale_layer(x_out)
            else:
                x_out = scale_layer(F.interpolate(x_out, size_int, mode='bilinear', align_corners=False))
                    
            # Append output of each scale
            self.output_list.append(x_out)

        # Loop ascending on resolution (start with lowest)
        k = self.scales
        for i in range(self.scales):

            # Identify scales and get correct size for interpolation
            scale_layer = self.scales_module[1+ i + self.scales]
            k -= 1 
            size_int = [int(j / 2**k) for j in list(x.size()[2:])]

            # First element does not have information from a lower resolution scale           
            if i == 0:
                x_out = scale_layer(F.interpolate(x_out, size_int, mode='bilinear', align_corners=False))
            else:
                x_out = scale_layer(torch.cat((F.interpolate(self.output_list[k], size_int, mode='bilinear', align_corners=False),
                                            F.interpolate(x_out, size_int, mode='bilinear', align_corners=False)), dim=1))

        # Final conv to output a single field
        x_out = out_conv(x_out)

        # Clear list to avoid memory leak
        self.output_list.clear()

        return x_out
