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

class MSNet3(BaseModel):
    """
    Original network. Define the network. The only input needed is the number of data (input) channels.
    Procedure:
        - Downsample input to quarter scale and use ConvBlock1.
        - Upsample output of ConvBlock1 to half scale.
        - Downsample input to half scale, concat with output of ConvBlock1, and use ConvBlock2.
        - Upsample output of ConvBlock2 to full scale.
        - Concat input and output of ConvBlock2, use ConvBlock3. Output of ConvBlock3 has 8 channels.
        - Use final Conv2d layer with kernel_size of 1 to go from 4 channels to 1 output channel.
    """
    def __init__(self, data_channels):
        super(MSNet3, self).__init__()
        self.conv_4 = _ConvBlock1(data_channels, 32, 64, 1)
        self.conv_2 = _ConvBlock2(data_channels + 1, 32, 64, 128, 1)
        self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        quarter_size = [int(i * 0.25) for i in list(x.size()[2:])]
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        conv_4_out = self.conv_4(F.interpolate(x, quarter_size, mode='bilinear'))
        conv_2_out = self.conv_2(torch.cat((F.interpolate(x, half_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_1_out = self.conv_1(torch.cat((F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False),
                                            F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)),
                                           dim=1))
        final_out = self.final(conv_1_out)
        return final_out

class MSNet4(BaseModel):
    """
    Class for 4 scales Multi-Scale network
    """
    def __init__(self, data_channels):
        super(MSNet4, self).__init__()
        self.conv_8 = _ConvBlock1(data_channels, 32, 64, 1)
        self.conv_4 = _ConvBlock1(data_channels + 1, 32, 64, 1)
        self.conv_2 = _ConvBlock2(data_channels + 1, 32, 64, 96, 1)
        self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        eight_size = [int(i / 8) for i in list(x.size()[2:])]
        quarter_size = [int(i * 0.25) for i in list(x.size()[2:])]
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        conv_8_out = self.conv_8(F.interpolate(x, eight_size, mode='bilinear'))
        conv_4_out = self.conv_4(torch.cat((F.interpolate(x, quarter_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_8_out, quarter_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_2_out = self.conv_2(torch.cat((F.interpolate(x, half_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_1_out = self.conv_1(torch.cat((F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False),
                                            F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)),
                                           dim=1))
        final_out = self.final(conv_1_out)
        return final_out

class MSNet5(BaseModel):
    """
    Class for 5 scales Multi-Scale network
    """
    def __init__(self, data_channels):
        super(MSNet5, self).__init__()
        self.conv_16 = _ConvBlock1(data_channels, 32, 64, 1)
        self.conv_8 = _ConvBlock1(data_channels + 1, 32, 64, 1)
        self.conv_4 = _ConvBlock1(data_channels + 1, 32, 64, 1)
        self.conv_2 = _ConvBlock2(data_channels + 1, 32, 48, 96, 1)
        self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        sixteen_size = [int(i / 16) for i in list(x.size()[2:])]
        eight_size = [int(i / 8) for i in list(x.size()[2:])]
        quarter_size = [int(i * 0.25) for i in list(x.size()[2:])]
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        conv_16_out = self.conv_16(F.interpolate(x, sixteen_size, mode='bilinear'))
        conv_8_out = self.conv_8(torch.cat((F.interpolate(x, eight_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_16_out, eight_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_4_out = self.conv_4(torch.cat((F.interpolate(x, quarter_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_8_out, quarter_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_2_out = self.conv_2(torch.cat((F.interpolate(x, half_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_1_out = self.conv_1(torch.cat((F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False),
                                            F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)),
                                           dim=1))
        final_out = self.final(conv_1_out)
        return final_out

class MSNet6(BaseModel):
    """
    Class for 6 scales Multi-Scale network
    """
    def __init__(self, data_channels):
        super(MSNet6, self).__init__()
        self.conv_32 = _ConvBlock1(data_channels, 16, 32, 1)
        self.conv_16 = _ConvBlock1(data_channels + 1, 32, 64, 1)
        self.conv_8 = _ConvBlock1(data_channels + 1, 32, 64, 1)
        self.conv_4 = _ConvBlock1(data_channels + 1, 32, 64, 1)
        self.conv_2 = _ConvBlock2(data_channels + 1, 32, 52, 96, 1)
        self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 112, 8)
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        thirty_two_size = [int(i / 32) for i in list(x.size()[2:])]
        sixteen_size = [int(i / 16) for i in list(x.size()[2:])]
        eight_size = [int(i / 8) for i in list(x.size()[2:])]
        quarter_size = [int(i * 0.25) for i in list(x.size()[2:])]
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        conv_32_out = self.conv_32(F.interpolate(x, thirty_two_size, mode='bilinear'))
        conv_16_out = self.conv_16(torch.cat((F.interpolate(x, sixteen_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_32_out, sixteen_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_8_out = self.conv_8(torch.cat((F.interpolate(x, eight_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_16_out, eight_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_4_out = self.conv_4(torch.cat((F.interpolate(x, quarter_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_8_out, quarter_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_2_out = self.conv_2(torch.cat((F.interpolate(x, half_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_1_out = self.conv_1(torch.cat((F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False),
                                            F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)),
                                           dim=1))
        final_out = self.final(conv_1_out)
        return final_out

class MSNet5_small(BaseModel):
    """
    Class for 5 scales Multi-Scale network with less parameters
    compared to MSNet5
    """
    def __init__(self, data_channels):
        super(MSNet5_small, self).__init__()
        self.conv_16 = _ConvBlock1(data_channels, 20, 40, 1)
        self.conv_8 = _ConvBlock1(data_channels + 1, 20, 40, 1)
        self.conv_4 = _ConvBlock1(data_channels + 1, 20, 40, 1)
        self.conv_2 = _ConvBlock2(data_channels + 1, 20, 40, 80, 1)
        self.conv_1 = _ConvBlock3(data_channels + 1, 20, 40, 80, 8)
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        sixteen_size = [int(i / 16) for i in list(x.size()[2:])]
        eight_size = [int(i / 8) for i in list(x.size()[2:])]
        quarter_size = [int(i * 0.25) for i in list(x.size()[2:])]
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        conv_16_out = self.conv_16(F.interpolate(x, sixteen_size, mode='bilinear'))
        conv_8_out = self.conv_8(torch.cat((F.interpolate(x, eight_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_16_out, eight_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_4_out = self.conv_4(torch.cat((F.interpolate(x, quarter_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_8_out, quarter_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_2_out = self.conv_2(torch.cat((F.interpolate(x, half_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_1_out = self.conv_1(torch.cat((F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False),
                                            F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)),
                                           dim=1))
        final_out = self.final(conv_1_out)
        return final_out

class MSNet5_big(BaseModel):
    """
    Class for 5 scales Multi-Scale network with more parameters
    compared to MSNet5
    """
    def __init__(self, data_channels):
        super(MSNet5_big, self).__init__()
        self.conv_16 = _ConvBlock1(data_channels, 32, 64, 1)
        self.conv_8 = _ConvBlock1(data_channels + 1, 50, 100, 1)
        self.conv_4 = _ConvBlock1(data_channels + 1, 54, 110, 1)
        self.conv_2 = _ConvBlock2(data_channels + 1, 32, 64, 112, 1)
        self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        sixteen_size = [int(i / 16) for i in list(x.size()[2:])]
        eight_size = [int(i / 8) for i in list(x.size()[2:])]
        quarter_size = [int(i * 0.25) for i in list(x.size()[2:])]
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        conv_16_out = self.conv_16(F.interpolate(x, sixteen_size, mode='bilinear'))
        conv_8_out = self.conv_8(torch.cat((F.interpolate(x, eight_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_16_out, eight_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_4_out = self.conv_4(torch.cat((F.interpolate(x, quarter_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_8_out, quarter_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_2_out = self.conv_2(torch.cat((F.interpolate(x, half_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_1_out = self.conv_1(torch.cat((F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False),
                                            F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)),
                                           dim=1))
        final_out = self.final(conv_1_out)
        return final_out

class MSNet5_big_1(BaseModel):
    """
    Class for 5 scales Multi-Scale network with added parameters in
    conv_4 and conv_8 compared to MSNet5_big
    """
    def __init__(self, data_channels):
        super(MSNet5_big_1, self).__init__()
        self.conv_16 = _ConvBlock1(data_channels, 32, 64, 1)
        self.conv_8 = _ConvBlock1(data_channels + 1, 96, 144, 1)
        self.conv_4 = _ConvBlock1(data_channels + 1, 96, 144, 1)
        self.conv_2 = _ConvBlock2(data_channels + 1, 32, 64, 112, 1)
        self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        sixteen_size = [int(i / 16) for i in list(x.size()[2:])]
        eight_size = [int(i / 8) for i in list(x.size()[2:])]
        quarter_size = [int(i * 0.25) for i in list(x.size()[2:])]
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        conv_16_out = self.conv_16(F.interpolate(x, sixteen_size, mode='bilinear'))
        conv_8_out = self.conv_8(torch.cat((F.interpolate(x, eight_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_16_out, eight_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_4_out = self.conv_4(torch.cat((F.interpolate(x, quarter_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_8_out, quarter_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_2_out = self.conv_2(torch.cat((F.interpolate(x, half_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_1_out = self.conv_1(torch.cat((F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False),
                                            F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)),
                                           dim=1))
        final_out = self.final(conv_1_out)
        return final_out

class MSNet5_big_2(BaseModel):
    """
    Class for 5 scales Multi-Scale network with added parameters in
    conv_1 and conv_2 compared to MSNet5_big
    """
    def __init__(self, data_channels):
        super(MSNet5_big_2, self).__init__()
        self.conv_16 = _ConvBlock1(data_channels, 32, 64, 1)
        self.conv_8 = _ConvBlock1(data_channels + 1, 50, 100, 1)
        self.conv_4 = _ConvBlock1(data_channels + 1, 54, 110, 1)
        self.conv_2 = _ConvBlock2(data_channels + 1, 48, 96, 128, 1)
        self.conv_1 = _ConvBlock3(data_channels + 1, 48, 96, 144, 8)
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        sixteen_size = [int(i / 16) for i in list(x.size()[2:])]
        eight_size = [int(i / 8) for i in list(x.size()[2:])]
        quarter_size = [int(i * 0.25) for i in list(x.size()[2:])]
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        conv_16_out = self.conv_16(F.interpolate(x, sixteen_size, mode='bilinear'))
        conv_8_out = self.conv_8(torch.cat((F.interpolate(x, eight_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_16_out, eight_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_4_out = self.conv_4(torch.cat((F.interpolate(x, quarter_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_8_out, quarter_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_2_out = self.conv_2(torch.cat((F.interpolate(x, half_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_1_out = self.conv_1(torch.cat((F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False),
                                            F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)),
                                           dim=1))
        final_out = self.final(conv_1_out)
        return final_out

class _ConvBlockShort(nn.Module):
    """
    First block - quarter scale.
    Four Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is the same as input size)
    Optional dropout before the final Conv2d layer.
    ReLU after the first two Conv2d layers, not after the last two - predictions can be positive or negative.
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels, out_channels, dropout=False):
        super(_ConvBlockShort, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=3, padding=0),
            nn.ReLU(),
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


class _ConvBlockLong(nn.Module):
    """
    Second block - half scale.
    Six Conv2d layers, the first one with kernel_size 5, padding 2 and the remainder with kernel_size 3 and padding 1.
    Optional dropout before the final Conv2d layer.
    ReLU after the first four Conv2d layers, not after the last two - predictions can be positive or negative.
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels, mid3_channels, out_channels, dropout=False):
        super(_ConvBlockLong, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=3, padding=0),
            nn.ReLU(),
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


class MSNet5_rev(BaseModel):
    """
    Class for 5 scales Multi-Scale network
    """
    def __init__(self, data_channels):
        super(MSNet5_rev, self).__init__()
        self.conv_16 = _ConvBlockLong(data_channels, 32, 64, 112, 1)
        self.conv_8 = _ConvBlockLong(data_channels + 1, 32, 64, 96, 1)
        self.conv_4 = _ConvBlockShort(data_channels + 1, 32, 64, 1)
        self.conv_2 = _ConvBlockShort(data_channels + 1, 32, 64, 1)
        self.conv_1 = _ConvBlockShort(data_channels + 1, 32, 64, 8)
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        sixteen_size = [int(i / 16) for i in list(x.size()[2:])]
        eight_size = [int(i / 8) for i in list(x.size()[2:])]
        quarter_size = [int(i * 0.25) for i in list(x.size()[2:])]
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        conv_16_out = self.conv_16(F.interpolate(x, sixteen_size, mode='bilinear'))
        conv_8_out = self.conv_8(torch.cat((F.interpolate(x, eight_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_16_out, eight_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_4_out = self.conv_4(torch.cat((F.interpolate(x, quarter_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_8_out, quarter_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_2_out = self.conv_2(torch.cat((F.interpolate(x, half_size, mode='bilinear', align_corners=False),
                                            F.interpolate(conv_4_out, half_size, mode='bilinear', align_corners=False)),
                                           dim=1))
        conv_1_out = self.conv_1(torch.cat((F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False),
                                            F.interpolate(conv_2_out, x.size()[2:], mode='bilinear', align_corners=False)),
                                           dim=1))
        final_out = self.final(conv_1_out)
        return final_out