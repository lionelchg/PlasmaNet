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


class _ConvBlock1(nn.Module):
    """
    First block - quarter scale.
    Four Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is the same as input size)
    Optional dropout before the final Conv2d layer.
    ReLU after the first two Conv2d layers, not after the last two - predictions can be positive or negative.
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels, out_channels, dropout=False):
        super(_ConvBlock1, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, mid2_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels, mid1_channels, kernel_size=3, padding=0),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, out_channels, kernel_size=3, padding=0)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock2(nn.Module):
    """
    Second block - half scale.
    Six Conv2d layers, the first one with kernel_size 5, padding 2 and the remainder with kernel_size 3 and padding 1.
    Optional dropout before the final Conv2d layer.
    ReLU after the first four Conv2d layers, not after the last two - predictions can be positive or negative.
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels, mid3_channels, out_channels, dropout=False):
        super(_ConvBlock2, self).__init__()
        layers = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, mid2_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels, mid3_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid3_channels, mid2_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels, mid1_channels, kernel_size=3, padding=0),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, out_channels, kernel_size=3, padding=0)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock3(nn.Module):
    """
    Third block - full scale.
    Six Conv2d layers, the first and last ones with kernel_size 5, padding 2 and the remainder with kernel_size 3 and
    padding 1.
    Optional dropout before the final Conv2d layer.
    ReLU after the first four Conv2d layers, not after the last two - predictions can be positive or negative.
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels, mid3_channels, out_channels, dropout=False):
        super(_ConvBlock3, self).__init__()
        layers = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, mid2_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels, mid3_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid3_channels, mid2_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels, mid1_channels, kernel_size=3, padding=0),
            nn.ReplicationPad2d(2),
            nn.Conv2d(mid1_channels, out_channels, kernel_size=5, padding=0)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class MultiSimpleNet(BaseModel):
    """
    Define the network. The only input needed is the number of data (input) channels.
    Procedure:
        - Downsample input to quarter scale and use ConvBlock1.
        - Upsample output of ConvBlock1 to half scale.
        - Downsample input to half scale, concat with output of ConvBlock1, and use ConvBlock2.
        - Upsample output of ConvBlock2 to full scale.
        - Concat input and output of ConvBlock2, use ConvBlock3. Output of ConvBlock3 has 8 channels.
        - Use final Conv2d layer with kernel_size of 1 to go from 8 channels to 1 output channel.
        - This same procedure is used to combine the exit of each layer, this new connexion should
            help to analyse the task specialization of each scale
    """
    def __init__(self, data_channels):
        super(MultiSimpleNet, self).__init__()
        self.conv_4 = _ConvBlock1(data_channels, 32, 64, 1)
        self.conv_2 = _ConvBlock2(data_channels + 1, 32, 64, 128, 1)
        self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
        self.final = nn.Conv2d(8, 1, kernel_size=1)
        self.w_s_4 = nn.Conv2d(1, 1, kernel_size=1)
        self.w_s_2 = nn.Conv2d(1, 1, kernel_size=1)
        self.w_s_1 = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x, epoch):
        quarter_size = [int(i * 0.25) for i in list(x.size()[2:])]
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        conv_4_out = self.conv_4(F.interpolate(x, quarter_size, mode='bilinear'))
        conv_2_out = self.conv_2(torch.cat((F.interpolate(x, half_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_1_out = self.conv_1(torch.cat((F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False),
                                            F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)),
                                           dim=1))

        #final_big = self.w_s_1(self.final(conv_1_out))
        #final_medium = self.w_s_2(F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False))
        #final_small = self.w_s_4( F.interpolate(F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False), x.size()[2:], mode='bilinear', align_corners=False))

        final_big = self.final(conv_1_out)
        final_medium = F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)
        final_small = F.interpolate(F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False), x.size()[2:], mode='bilinear', align_corners=False)

        small_val =1.0
        med_val = 1.0
        big_val = 1.0

        #if epoch > 100:
        #    med_val = 1.0
        #if epoch > 300:
        #    big_val = 1.0
        
        final_out = big_val * final_big + med_val * final_medium + small_val * final_small
        #final_out = final_big + final_medium + final_small
        #final_out = final_big

        output_fields = torch.cat((final_out, final_big, final_medium, final_small), dim=1)
        return output_fields
