import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
import random

#Create the model

class _ConvBlock1(nn.Module):
    """
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock1, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
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
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
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
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock7(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock7, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlock8(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock8, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlock9(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock9, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)          

class _ConvBlock10(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock10, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class UNet(nn.Module):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 4 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self,data_channels):
        super(UNet, self).__init__()
        self.convN_1 = _ConvBlock1(1,32)
        self.convN_2 = _ConvBlock2(32, 32)
        self.convN_3 = _ConvBlock3(32,32)
        self.convN_4 = _ConvBlock4(32,64)
        self.convN_5 = _ConvBlock5(64,64)
        self.convN_6 = _ConvBlock6(128,64)
        self.convN_7 = _ConvBlock7(96,64)
        self.convN_8 = _ConvBlock8(96,32)
        self.convN_9 = _ConvBlock9(64,32)
        self.final = _ConvBlock10(32,1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(convN_3out)
        convN_5out = self.convN_5(convN_4out)
        convN_6out = self.convN_6( torch.cat((F.upsample(convN_5out,scale_factor=2,mode = 'bilinear'), convN_4out),dim = 1) )
        convN_7out = self.convN_7( torch.cat((F.upsample(convN_6out,scale_factor=2,mode = 'bilinear'), convN_3out),dim = 1) )
        convN_8out = self.convN_8( torch.cat((F.upsample(convN_7out,scale_factor=2,mode = 'bilinear'), convN_2out),dim = 1) )
        convN_9out = self.convN_9( torch.cat((F.upsample(convN_8out,scale_factor=2,mode = 'bilinear'), convN_1out),dim = 1) )
        final_out = self.final(convN_9out)
        return final_out
