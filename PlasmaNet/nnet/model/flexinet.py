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


class _ConvscaleFlexible(nn.Module):
    """
    Flexible convolutional layer.
    The network will have conv2D layers with an intermediate activation function at each scale.
    Each layer will have a kernel of size 3x3 and no stride. Padding is done with zero padding
    (default behavior of nn.Conv2d).
    """
    def __init__(self, data_channels, filters, pad_method):
        """ Initialization of the convolutional layer
        The order is: Pad Conv (ReLu Pad Conv) * len(filters).

        :param data_channels: number of input images
        :type data_channels: int
        :param filters: number of filters in the network's layers
        :type filters: list
        :param pad_method: padding mode, to choose between 'circular', 'zeros', 'reflect', 'replicate'
        :type filters: str
        """
        super(_ConvscaleFlexible, self).__init__()

        # Module list containing all the layers
        self.scale_filters = nn.ModuleList()

        # Initialize with input size.
        self.scale_filters.append(nn.Conv2d(data_channels, filters[0], kernel_size=3, stride=1, padding=1, padding_mode=pad_method))

        # Intermediate layers.
        for i in range(len(filters)-2):
            self.scale_filters.append(nn.ReLU())
            self.scale_filters.append(nn.Conv2d(filters[i], filters[i+1], kernel_size=3, stride=1, padding=1, padding_mode=pad_method))

        if len(filters) > 1:
            # Final layer, note that there is no ReLU at the end.
            self.scale_filters.append(nn.ReLU())
            self.scale_filters.append(nn.Conv2d(filters[-2], filters[-1], kernel_size=3, stride=1, padding=1, padding_mode=pad_method))

    def forward(self, x):
        for i in range(len(self.scale_filters)):
            x = self.scale_filters[i](x)
        return x


class FlexiNet(nn.Module):
    """
    Define the network with *data_channels* number of inputs while *filters* hold the size of 
    all the convolutional layers applied. The first element of the list will corrrespond to the lowest resolution scale, 
    whereas the last one will be the scale with the original size.

    The number of scales MUST correspond to the number of lists in the filters list.

    The network is supposed symmetric, so the conv blocks of a same scale are equal, and
    have as many conv2D layers with an intermediate activation function at each scale as indicated on the
    scales lists.
    The order is: Pad Conv ReLu Pad Conv
    Depending on the number of scales added, the network will consequently interpolate the domain as input.
    All the outputs will be added at the end.
    """
    def __init__(self, data_channels, filters, scales, input_val, output_val, pad_method='zeros'):
        """ Initialization of FlexiNet with data_channels in in

        :param data_channels: The nunmber of data (input) channels
        :type data_channels: int
        :param filters: List of integers for the number of filters of each layer
        :type filters: list
        :param scales: number of scales
        :type scales: int
        :param input_val: boolean to check if the input is forwarded to all the blocks.
        :type input_val: bool
        :param output_val: boolean to check if the output of each scale is added at the end.
        :type output_val: bool
        :param pad_method: padding mode, to choose between 'circular', 'zeros', 'reflect', 'replicate'
        :type filters: str (optional, default = zeros)
        """
        super(FlexiNet, self).__init__()

        self.scales = scales
        self.input_val = input_val
        self.output_val = output_val
        self.scales_module = nn.ModuleList()
        self.pad_method = pad_method
        self.output_list = []

        assert len(filters) == scales, 'Scales and inputted filters do not match'

        # Initialize highest resolution scale
        self.scales_module.append(_ConvscaleFlexible(data_channels, filters[0], self.pad_method))

        # Loop to append scales from highest to lowest resolution
        for i in range(len(filters)-1):
            concat_input = filters[i][-1]
            if self.input_val:
                self.scales_module.append(_ConvscaleFlexible(data_channels + concat_input,
                                                             filters[i + 1], self.pad_method))
            else:
                self.scales_module.append(_ConvscaleFlexible(concat_input, filters[i + 1], self.pad_method))

        # The lowest resolution scale will have two blocks as well (even if not necessary)
        # to ensure a more proportional weight distribution
        self.scales_module.append(_ConvscaleFlexible(filters[-1][-1], filters[-1], self.pad_method))

        # Descending loop where scales are added increasing the resolution
        for i in range(len(filters) - 2, -1, -1):
            concat_input = filters[i][-1] + filters[i + 1][-1]
            self.scales_module.append(_ConvscaleFlexible(concat_input, filters[i], self.pad_method))

        # Final layer to return an output of size 1, when resolution is at its highest
        self.final_block = _ConvscaleFlexible(filters[0][-1], [1], self.pad_method)

    def forward(self, x):
        """
        Inputs:
        - x (torch.tensor): Tensor containing input fields (bsz, channels, h, w)
        """
        # Initialize output of each scale
        if torch.cuda.is_available():
            x_layers_out = torch.zeros((x.size(0), self.scales, 1, x.size(2), x.size(3))).cuda()
        else:
            x_layers_out = torch.zeros((x.size(0), self.scales, 1, x.size(2), x.size(3))) 

        # Loop descending on resolution (start with highest)
        for i in range(self.scales):

            # Identify scale and get the size to interpolate
            scale_layer = self.scales_module[i]
            size_int = [int(j / 2**i) for j in list(x.size()[2:])]

            # Highest resolution scale only takes input
            if i == 0:
                x_out = scale_layer(x)
            else:
                # Depending if input is added at different scales or not
                if self.input_val:
                    x_out = scale_layer(torch.cat((
                        F.interpolate(x, size_int, mode='bilinear', align_corners=False),
                        F.interpolate(x_out, size_int, mode='bilinear', align_corners=False),
                    ), dim=1))
                else:
                    x_out = scale_layer(F.interpolate(x_out, size_int, mode='bilinear', align_corners=False))
                    
            # Append output of each scale
            self.output_list.append(x_out)

        # Loop ascending on resolution (start with lowest)
        k = self.scales
        for i in range(self.scales):

            # Identify scales and get correct size for interpolation
            scale_layer = self.scales_module[i + self.scales]
            k -= 1 
            size_int = [int(j / 2**k) for j in list(x.size()[2:])]

            # First element does not have information from a lower resolution scale           
            if i == 0:
                x_out = scale_layer(F.interpolate(x_out, size_int, mode='bilinear', align_corners=False))
            else:
                x_out = scale_layer(torch.cat((
                    F.interpolate(self.output_list[k], size_int, mode='bilinear', align_corners=False),
                    F.interpolate(x_out, size_int, mode='bilinear', align_corners=False)
                ), dim=1))
            if self.output_val:    
                # Last element of layer is saved!
                x_layers_out[:, i] = F.interpolate(x_out[:, 0].unsqueeze(1), x[0, 0].shape,
                                                   mode='bilinear', align_corners=False)

        # Final conv to output a single field
        x_out = self.final_block(x_out)

        if self.output_val:
            x_layers_out[:, -1] = x_out
            for i in range(self.scales-1):
                x_out += x_layers_out[:, i]

        # Clear list to avoid memory leak
        self.output_list.clear()

        return x_out
