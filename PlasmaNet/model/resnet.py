########################################################################################################################
#                                                                                                                      #
#                                      ResNet (similar to ResNet18 but not quite)                                      #
#                                                                                                                      #
#                                               Ekhi Ajuria,CERFACS, 07.09.2020                                        #
#                                                                                                                      #
########################################################################################################################

"""
ResNet network

Inputs are shape (batch, channels, height, width)
Outputs are shape (batch,1, height, width)

The number of input (data) channels is selected when the model is created.
the number of output (target) channels is fixed at 1, although this could be changed in the future.

The data can be any size (i.e. height and width).

The model can be trained on data of a given size (H and W) and then used on data of any other size,
although the best results so far have been obtained with test data of similar size to the training data

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
import random

from ..base import BaseModel

# Create the model


class _ConvBlock_Basic(nn.Module):
    """
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock_Basic, self).__init__()
        layers = [
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock_Init(nn.Module):
    """
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU only after the first Conv2d layers
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock_Init, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock_Out(nn.Module):
    """
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU only after the first Conv2d layers
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock_Out, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class ResNet(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 4 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """

    def __init__(self, data_channels):
        super(ResNet, self).__init__()
        self.convN_1 = _ConvBlock_Init(data_channels, 32)
        self.convN_2 = _ConvBlock_Basic(32, 32)
        self.convN_3 = _ConvBlock_Basic(32, 64)
        self.convN_4 = _ConvBlock_Basic(64, 64)
        self.convN_5 = _ConvBlock_Basic(64, 64)
        self.convN_6 = _ConvBlock_Basic(64, 64)
        self.convN_7 = _ConvBlock_Basic(64, 64)
        self.convN_8 = _ConvBlock_Basic(64, 32)
        self.convN_9 = _ConvBlock_Basic(64, 32)
        self.final = _ConvBlock_Out(32, 1)

    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out+convN_1out)
        convN_4out = self.convN_4(
            convN_3out+torch.cat((convN_2out, torch.zeros_like(convN_2out)), dim=1))
        convN_5out = self.convN_5(convN_4out+convN_3out)
        convN_6out = self.convN_6(convN_5out+convN_4out)
        #convN_6out = self.convN_6(convN_5out+torch.cat((convN_4out,torch.zeros_like(convN_4out)),dim=1))
        convN_7out = self.convN_7(convN_6out+convN_5out)
        convN_8out = self.convN_8(convN_7out+convN_6out)
        convN_9out = self.convN_9(
            torch.cat((convN_8out, torch.zeros_like(convN_8out)), dim=1)+convN_7out)
        #convN_9out = self.convN_9(convN_8out+convN_7out)
        final_out = self.final(convN_9out)
        return final_out
