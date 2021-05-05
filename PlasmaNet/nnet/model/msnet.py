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

from .scalesnet import ScalesNet

class _ConvBlock(nn.Module):
    """
    General convolution block for UNet. Depending on the location of the block
    in the architecture, the block can begin with a MaxPool2d (for bottom)
    or end with an UpSample or deconvolution layer (for up)
    """
    def __init__(self, fmaps, in_size, out_size, block_type, kernel_size, 
            pad_method='zeros'):
        super(_ConvBlock, self).__init__()
        layers = list()
        # Apply downsampling in middle blocks
        if block_type == 'middle':
            layers.append(nn.Upsample(in_size, mode='bilinear'))

        # Append all the specified layers
        for i in range(len(fmaps) - 1):
            layers.append(nn.Conv2d(fmaps[i], fmaps[i + 1], 
                kernel_size=kernel_size, padding=int((kernel_size - 1) / 2),
                padding_mode=pad_method))
            # No ReLu at the very last layer
            if i != len(fmaps) - 2 or block_type != 'out':
                layers.append(nn.ReLU())

        # Apply either Upsample or deconvolution
        if block_type == 'middle':
            layers.append(nn.Upsample(out_size, mode='bilinear'))

        # Build the sequence of layers
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class MSNet(ScalesNet):
    """
    General MSNet. All the layers are specified in the config file. Three different options are possible
    when going up the U: upsample, deconvolution or interpolation. Only interpolation
    allows the network to work on different resolutions
    """
    def __init__(self, scales, kernel_sizes, input_res):
        super(MSNet, self).__init__(scales, kernel_sizes)
        # For upsample the list of resolution is needed when 
        # the number of points is not a power of 2
        list_res = [int(input_res / 2**i) for i in range(self.n_scales)]

        # create down_blocks, bottom_fmaps and up_blocks
        middle_blocks = list()
        for local_depth in range(self.depth):
            middle_blocks.append(self.scales[f'depth_{self.depth - local_depth:d}'])
        out_fmaps = self.scales['depth_0']

        # Intemediate layers up (UpSample/Deconv at the end)
        self.ConvsUp = nn.ModuleList()
        for imiddle, middle_fmaps in enumerate(middle_blocks):
            self.ConvsUp.append(_ConvBlock(middle_fmaps, 
                in_size=list_res[-1 -imiddle], out_size=list_res[-2 -imiddle], 
                block_type='middle', kernel_size=self.kernel_sizes[-1 - imiddle]))
        
        # Out layer
        self.ConvsUp.append(_ConvBlock(out_fmaps, 
            in_size=list_res[0], out_size=list_res[0],
            block_type='out', kernel_size=self.kernel_sizes[0]))

    def forward(self, x):
        initial_map = x        
        # Apply the up loop
        for iconv, ConvUp in enumerate(self.ConvsUp):
            # First layer of convolution doesn't need concatenation
            if iconv == 0:
                x = ConvUp(x)
            else:
                x = ConvUp(torch.cat((x, initial_map), dim=1))
                
        return x









