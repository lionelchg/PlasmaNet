import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
import random

from ..base import BaseModel

# Create the model

class _ConvBlock1(nn.Module):
    """
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock1, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlock2(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock2, self).__init__()
        layers = [
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlock3(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock3, self).__init__()
        layers = [
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlock4(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock4, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)



class _ConvBlock5(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock5, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlock6(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock6, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class UNet3(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 3 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(UNet3, self).__init__()
        self.convN_1 = _ConvBlock1(1, 64)
        self.convN_2 = _ConvBlock2(64, 64)
        self.convN_3 = _ConvBlock3(64, 64)
        self.convN_4 = _ConvBlock4(128, 64)
        self.convN_5 = _ConvBlock5(128, 64)
        self.final = _ConvBlock6(64, 1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(torch.cat((F.interpolate(convN_3out, size=convN_2out[0,0].shape, mode='bilinear', align_corners= False), convN_2out), dim=1))
        convN_5out = self.convN_5(torch.cat((F.interpolate(convN_4out, size=convN_1out[0,0].shape, mode='bilinear', align_corners=False), convN_1out), dim=1))
        final_out = self.final(convN_5out)
        return final_out

        #F.interpolate(x, half_size, mode='bilinear', align_corners=False)
