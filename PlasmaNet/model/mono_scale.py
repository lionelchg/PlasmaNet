########################################################################################################################
#                                                                                                                      #
#                           MonoScale: CNN similar to the MS, but only with a single scale                             #
#                                                                                                                      #
#                                               Ekhi Ajuria,CERFACS, 07.09.2020                                        #
#                                                                                                                      #
########################################################################################################################

"""
MonoScale network

Inputs are shape (batch, channels, height, width)
Outputs are shape (batch,1, height, width)

The number of input (data) channels is selected when the model is created.
the number of output (target) channels is fixed at 1, although this could be changed in the future.

The data can be any size (i.e. height and width), although for best results the height and width should
be divisble by four.

The model can be trained on data of a given size (H and W) and then used on data of any other size,
although the best results so far have been obtained with test data of similar size to the training data

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """
    Full scale.
    Eight Conv2d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv2d layer
    ReLU after first four Conv2d layers, not after last - predictions can be +ve or -ve
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels,mid3_channels, mid4_channels,out_channels,dropout=False):
        super(_ConvBlock, self).__init__()
        layers = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, mid2_channels, kernel_size=3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels,mid3_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid3_channels,mid4_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid4_channels,mid3_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid3_channels,mid2_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels,mid1_channels,kernel_size = 3),
            nn.ReLU(),
        ]
        layers.append(nn.ReplicationPad2d(1))
        layers.append(nn.Conv2d(mid1_channels, out_channels, kernel_size = 3))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class MonoScale(nn.Module):
    """
    Define the network. Only input when called is number of data (input) channels.
        -Use final Conv2d layer with kernel size of 1 to go from 8 channels to 1 output channel.
    """
    def __init__(self,data_channels):
        super(MonoScale, self).__init__()
        self.convN = _ConvBlock(data_channels, 32,32,128,128,8)
        self.final = nn.Conv2d(8,1, kernel_size = 1)

    def forward(self, x):

        convN_out = self.convN(x)
        final_out = self.final(convN_out)

        return final_out
