import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
import random

from ..base import BaseModel

# Create the model

class _ConvBlock(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, fmaps, block_type):
        super(_ConvBlock, self).__init__()
        layers = list()
        # Apply pooling depending on pool boolean
        if block_type == 'down' or block_type == 'bottom':
            layers.append(nn.MaxPool2d(2))

        # Append all the specified layers
        for i in range(len(fmaps) - 1):
            layers.append(nn.Conv2d(fmaps[i], fmaps[i + 1], kernel_size=3, padding=1))
            layers.append(nn.ReLU())

        if block_type == 'up' or block_type == 'bottom':
            # layers.append(nn.ConvTranspose2d(fmaps[-1], fmaps[-1], 2, stride=2))
            layers.append(nn.Upsample(scale_factor=2.0))
        
        # Build the sequence of layers
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class UNet(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 4 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, in_fmaps, down_blocks, bottom_fmaps, up_blocks, out_fmaps):
        super(UNet, self).__init__()
        self.ConvsDown = list()
        # Entry layer
        self.ConvsDown.append(_ConvBlock(in_fmaps, ''))

        # Intermediate down layers (with MaxPool at the beginning)
        for down_fmaps in down_blocks:
            self.ConvsDown.append(_ConvBlock(down_fmaps, 'down'))

        # Bottom layer (MaxPool at the beginning and Upsample/Deconv at the end)
        self.ConvBottom = _ConvBlock(bottom_fmaps, 'bottom')

        # Intemediate layers up (UpSample/Deconv at the end)
        self.ConvsUp = list()
        for up_fmaps in up_blocks:
            self.ConvsUp.append(_ConvBlock(up_fmaps, 'up'))
        
        # Out layer
        self.ConvsUp.append(_ConvBlock(out_fmaps, ''))
        
    def forward(self, x):
        # List of the temporary x that are used for linking with the up branch
        inputs_down = list()

        # Apply the down loop
        for ConvDown in self.ConvsDown:
            x = ConvDown(x)
            inputs_down.append(x)
        
        # Bottom part of the U
        x = self.ConvBottom(x)
        
        # Apply the up loop
        for ConvUp in self.ConvsUp:
            x = ConvUp(torch.cat(x, inputs_down.pop()))
        
