########################################################################################################################
#                                                                                                                      #
#                           MultiScale: neural network from the summer 2019 plasma workshop                            #
#                                                                                                                      #
#                        Ekhi Ajuria, Guillaume Bogopolsky (transcription) CERFACS, 26.02.2020                         #
#                                                                                                                      #
########################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class _ConvBlock(nn.Module):
    """
    MSNet convolutional layer.
    The network will have conv2D layers with an intermediate ReLU activation function at each scale.
    Each layer will have a kernel of size 3x3 and no stride. Padding method is chosen when initializing
    """
    def __init__(self, filters, kernels, pad_method='zeros'):
        """Generic scale, only needs the number of feature maps and kernel sizes.

        Args:
            filters (list): list containing the size of the intermediate feature maps
            kernels (list): list containing the initial and final kernel sizes (size=2)
            pad_method (str): padding method, 'zeros' by default.
        """
        super(_ConvBlock, self).__init__()

        # Module list containing all the layers
        self.scale_filters = []

        # Initialize with input size.
        self.scale_filters.append(nn.Conv2d(filters[0], filters[1], kernel_size=kernels[0], stride=1, padding=kernels[0]//2, padding_mode=pad_method))

        # Intermediate layers.
        for i in range(1, len(filters)-2):
            self.scale_filters.append(nn.ReLU())
            self.scale_filters.append(nn.Conv2d(filters[i], filters[i+1], kernel_size=3, stride=1, padding=1, padding_mode=pad_method))

        if len(filters) > 1:
            # Final layer, note that there is no ReLU before and after.
            self.scale_filters.append(nn.Conv2d(filters[-2], filters[-1], kernel_size=kernels[-1], stride=1, padding=kernels[-1]//2, padding_mode=pad_method))

        # Build the sequence of layers
        self.encode = nn.Sequential(*self.scale_filters)

    def forward(self, x):
        return self.encode(x)


class MSNet(BaseModel):
    """
    Generic MSNet network definition.
    """
    def __init__(self, filters, kernels, pad_method='zeros'):
        """ The following args are needed when intializing:

        Args:
            filters (list): List of lists containing number of FM, i.e., [[1, 32, 32, 1], [2, 32, 32, 1]]
            kernels (list): List of lists containing the first and last kernel sizes (as they can be 5 for the MSNet)
            i.e.= [[5,3], [5,5]]
            pad_method (str, optional): Defaults to 'zeros'.
        """

        super(MSNet, self).__init__()

        self.pad_method = pad_method
        self.scales = len(filters)
        self.scales_list = nn.ModuleList()
        self.final = nn.Conv2d(filters[-1][-1], 1, kernel_size=1)

        for i in range(self.scales):
            self.scales_list.append(_ConvBlock(filters[i], kernels[i], self.pad_method))


    def forward(self, x):

        # Loop from lowest to highest resolution
        for i in range(self.scales):
            # Identify scales and get correct size for interpolation
            scale_layer = self.scales_list[i]
            size_int = [int(j / 2**(self.scales-1-i)) for j in list(x.size()[2:])]

            # First element does not have information from a lower resolution scale           
            if i == 0:
                x_out = scale_layer(F.interpolate(x, size_int, mode='bilinear', align_corners=False))
            else:
                x_out = scale_layer(torch.cat((F.interpolate(x, size_int, mode='bilinear', align_corners=False),
                                            F.interpolate(x_out, size_int, mode='bilinear', align_corners=False)), dim=1))

        # Final conv to output a single field
        x = self.final(x_out)

        return x











