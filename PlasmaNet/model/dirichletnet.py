########################################################################################################################
#                                                                                                                      #
#                               DirichletNet: neural network to recreate fields from BC                                #
#                                                                                                                      #
#                        Ekhi Ajuria,  Guillaume Bogopolsky (transcription) CERFACS, 07.04.2020                        #
#                                                                                                                      #
########################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class _ConvBlock1(nn.Module):
    """
    First block - 1D Convolutions.
    Four Conv1d layers, with kernel_sizes 3,5,7,5 and padding of 1,2,3,1 (padding ensures output size is the same as input size) (Replication padding)
    Optional dropout before the final Conv2d layer.
    ReLU after the first two Conv1d layers, Leaky ReLu on the last two (slope of 0.1)- predictions can be positive or negat<Undo>ive.
    """
    def __init__(self, channels1, channels2, channels3, out_channels,dropout=False):
        super(_ConvBlock1, self).__init__()
        layers = [
            nn.ReplicationPad1d(1),
            nn.Conv1d(1, channels1, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad1d(2),
            nn.Conv1d(channels1, channels2, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.ReplicationPad1d(3),
            nn.Conv1d(channels2, channels3, kernel_size=7, padding=0),
            nn.LeakyReLU(0.1),
            nn.ReplicationPad1d(2),
            nn.Conv1d(channels3, out_channels, kernel_size=5, padding=0),
            nn.LeakyReLU(0.1)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock2(nn.Module):
    """
    Second block - 2D Convolutions.
    4 Conv2d layers, the first three with kernel_size 3, padding 1 and the remainder with kernel_size 5 and padding 2.(Replication padding)
    Optional dropout before the final Conv2d layer.
    ReLU after the first two Conv2d layers, LeakyReLU(0.1) at the third and nothing after the last - predictions can be positive or negative.
    """
    def __init__(self, channels1, channels2, channels3, out_channels, dropout=False):
        super(_ConvBlock2, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, channels1, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(channels1,channels2, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(channels2,channels3, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
            nn.ReplicationPad2d(2),
            nn.Conv2d(channels3, out_channels, kernel_size=5, padding=0)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class DirichletNet(BaseModel):
    """
    Define the network. Takes data_channels as input, although not used (easier object initialization)
    TODO change this
    Takes two inputs, x (BC tensor) and y (int for the length of opposite direction)
    Procedure:
        - Make a series of 1D convolutions on the BC input (array of lenght [bsz, 1, 1, N])
        - Transform [bsz, 64, 1, N] into [bsz, 1, 64, N]
        - Interpolate into size [ bsz, 1, y, N]
        - Perform a series of 2D Convolution until the final output is reached [bsz,1, N, N])
    """
    def __init__(self,data_channels):
        super(DirichletNet, self).__init__()
        self.conv_2 = _ConvBlock1(32, 64, 128, 64)
        self.conv_1 = _ConvBlock2(1, 16, 32, 16, 1)

    def forward(self, x):

        assert x.size(1) == 1 and x.size(2) == 1, "Input array does not have the size (bsz,1,1,N)"
        N = x.size(3)

        conv_2_out = self.conv_2(x[:,:,0,:]).unsqueeze(1)
        final_input = F.interpolate(conv_2_out, size=[N,N], mode='bilinear', align_corners=False)
        final_out = self.conv_1(final_input)

        return final_out
