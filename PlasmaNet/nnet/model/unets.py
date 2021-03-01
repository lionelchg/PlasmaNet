import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
import random

from ..base import BaseModel

# Create the model

class _ConvBlockIn(nn.Module):
    """
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_ConvBlockIn, self).__init__()
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlockDown(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_ConvBlockDown, self).__init__()
        layers = [
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlockUp(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_ConvBlockUp, self).__init__()
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlockOut(nn.Module):
    """
    Maxpooling to reduce the size
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU after first two Conv2d layers
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlockOut, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class UNet3(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 4 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(UNet3, self).__init__()
        self.convN_1 = _ConvBlockIn(1, 64, 64)
        self.convN_2 = _ConvBlockDown(64, 64, 64)
        self.convN_3 = _ConvBlockDown(64, 76, 64)
        self.convN_4 = _ConvBlockUp(128, 64, 64)
        self.convN_5 = _ConvBlockUp(128, 64, 64)
        self.final = _ConvBlockOut(64, 1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(torch.cat((F.interpolate(convN_3out, size=convN_2out[0, 0].shape, 
                                        mode='bilinear', align_corners= False), convN_2out), dim=1))
        convN_5out = self.convN_5(torch.cat((F.interpolate(convN_4out, size=convN_1out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_1out), dim=1))

        final_out = self.final(convN_5out)
        return final_out

class UNet4(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 4 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(UNet4, self).__init__()
        self.convN_1 = _ConvBlockIn(1, 48, 48)
        self.convN_2 = _ConvBlockDown(48, 48, 48)
        self.convN_3 = _ConvBlockDown(48, 48, 48)
        self.convN_4 = _ConvBlockDown(48, 64, 64)
        self.convN_5 = _ConvBlockUp(112, 64, 64)
        self.convN_6 = _ConvBlockUp(112, 60, 48)
        self.convN_7 = _ConvBlockUp(96, 48, 48)
        self.final = _ConvBlockOut(48, 1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(convN_3out)
        convN_5out = self.convN_5(torch.cat((F.interpolate(convN_4out, size=convN_3out[0, 0].shape, 
                                        mode='bilinear', align_corners= False), convN_3out), dim=1))
        convN_6out = self.convN_6(torch.cat((F.interpolate(convN_5out, size=convN_2out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_2out), dim=1))
        convN_7out = self.convN_7(torch.cat((F.interpolate(convN_6out, size=convN_1out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_1out), dim=1))
        final_out = self.final(convN_7out)
        return final_out

class UNet5(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 4 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(UNet5, self).__init__()
        self.convN_1 = _ConvBlockIn(1, 32, 32)
        self.convN_2 = _ConvBlockDown(32, 32, 32)
        self.convN_3 = _ConvBlockDown(32, 32, 32)
        self.convN_4 = _ConvBlockDown(32, 48, 62)
        self.convN_5 = _ConvBlockDown(62, 60, 62)
        self.convN_6 = _ConvBlockUp(124, 64, 64)
        self.convN_7 = _ConvBlockUp(96, 64, 64)
        self.convN_8 = _ConvBlockUp(96, 32, 32)
        self.convN_9 = _ConvBlockUp(64, 32, 32)
        self.final = _ConvBlockOut(32, 1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(convN_3out)
        convN_5out = self.convN_5(convN_4out)
        convN_6out = self.convN_6(torch.cat((F.interpolate(convN_5out, size=convN_4out[0, 0].shape, 
                                        mode='bilinear', align_corners= False), convN_4out), dim=1))
        convN_7out = self.convN_7(torch.cat((F.interpolate(convN_6out, size=convN_3out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_3out), dim=1))
        convN_8out = self.convN_8(torch.cat((F.interpolate(convN_7out, size=convN_2out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_2out), dim=1))
        convN_9out = self.convN_9(torch.cat((F.interpolate(convN_8out, size=convN_1out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_1out), dim=1))
        final_out = self.final(convN_9out)
        return final_out

class UNet5_small(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 4 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(UNet5_small, self).__init__()
        self.convN_1 = _ConvBlockIn(1, 20, 20)
        self.convN_2 = _ConvBlockDown(20, 20, 20)
        self.convN_3 = _ConvBlockDown(20, 20, 20)
        self.convN_4 = _ConvBlockDown(20, 48, 62)
        self.convN_5 = _ConvBlockDown(62, 60, 62)
        self.convN_6 = _ConvBlockUp(124, 64, 64)
        self.convN_7 = _ConvBlockUp(84, 64, 64)
        self.convN_8 = _ConvBlockUp(84, 20, 20)
        self.convN_9 = _ConvBlockUp(40, 20, 20)
        self.final = _ConvBlockOut(20, 1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(convN_3out)
        convN_5out = self.convN_5(convN_4out)
        convN_6out = self.convN_6(torch.cat((F.interpolate(convN_5out, size=convN_4out[0, 0].shape, 
                                        mode='bilinear', align_corners= False), convN_4out), dim=1))
        convN_7out = self.convN_7(torch.cat((F.interpolate(convN_6out, size=convN_3out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_3out), dim=1))
        convN_8out = self.convN_8(torch.cat((F.interpolate(convN_7out, size=convN_2out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_2out), dim=1))
        convN_9out = self.convN_9(torch.cat((F.interpolate(convN_8out, size=convN_1out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_1out), dim=1))
        final_out = self.final(convN_9out)
        return final_out

class UNet5_small_1(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 4 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(UNet5_small_1, self).__init__()
        self.convN_1 = _ConvBlockIn(1, 20, 20)
        self.convN_2 = _ConvBlockDown(20, 20, 20)
        self.convN_3 = _ConvBlockDown(20, 20, 20)
        self.convN_4 = _ConvBlockDown(20, 40, 40)
        self.convN_5 = _ConvBlockDown(40, 40, 40)
        self.convN_6 = _ConvBlockUp(80, 40, 40)
        self.convN_7 = _ConvBlockUp(60, 40, 40)
        self.convN_8 = _ConvBlockUp(60, 20, 20)
        self.convN_9 = _ConvBlockUp(40, 20, 20)
        self.final = _ConvBlockOut(20, 1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(convN_3out)
        convN_5out = self.convN_5(convN_4out)
        convN_6out = self.convN_6(torch.cat((F.interpolate(convN_5out, size=convN_4out[0, 0].shape, 
                                        mode='bilinear', align_corners= False), convN_4out), dim=1))
        convN_7out = self.convN_7(torch.cat((F.interpolate(convN_6out, size=convN_3out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_3out), dim=1))
        convN_8out = self.convN_8(torch.cat((F.interpolate(convN_7out, size=convN_2out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_2out), dim=1))
        convN_9out = self.convN_9(torch.cat((F.interpolate(convN_8out, size=convN_1out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_1out), dim=1))
        final_out = self.final(convN_9out)
        return final_out

class UNet5_big(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 4 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(UNet5_big, self).__init__()
        self.convN_1 = _ConvBlockIn(1, 40, 40)
        self.convN_2 = _ConvBlockDown(40, 40, 40)
        self.convN_3 = _ConvBlockDown(40, 40, 40)
        self.convN_4 = _ConvBlockDown(40, 48, 78)
        self.convN_5 = _ConvBlockDown(78, 78, 78)
        self.convN_6 = _ConvBlockUp(156, 78, 78)
        self.convN_7 = _ConvBlockUp(118, 78, 78)
        self.convN_8 = _ConvBlockUp(118, 40, 40)
        self.convN_9 = _ConvBlockUp(80, 40, 40)
        self.final = _ConvBlockOut(40, 1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(convN_3out)
        convN_5out = self.convN_5(convN_4out)
        convN_6out = self.convN_6(torch.cat((F.interpolate(convN_5out, size=convN_4out[0, 0].shape, 
                                        mode='bilinear', align_corners= False), convN_4out), dim=1))
        convN_7out = self.convN_7(torch.cat((F.interpolate(convN_6out, size=convN_3out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_3out), dim=1))
        convN_8out = self.convN_8(torch.cat((F.interpolate(convN_7out, size=convN_2out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_2out), dim=1))
        convN_9out = self.convN_9(torch.cat((F.interpolate(convN_8out, size=convN_1out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_1out), dim=1))
        final_out = self.final(convN_9out)
        return final_out

class UNet6(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
        - Perform 4 levels of convolution
        - When returning to the original size, concatenate output of matching sizes
        - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(UNet6, self).__init__()
        self.convN_1 = _ConvBlockIn(1, 32, 32)
        self.convN_2 = _ConvBlockDown(32, 32, 32)
        self.convN_3 = _ConvBlockDown(32, 32, 32)
        self.convN_4 = _ConvBlockDown(32, 48, 48)
        self.convN_5 = _ConvBlockDown(48, 48, 48)
        self.convN_6 = _ConvBlockDown(48, 48, 48)
        self.convN_7 = _ConvBlockUp(96, 60, 48)
        self.convN_8 = _ConvBlockUp(96, 48, 48)
        self.convN_9 = _ConvBlockUp(80, 48, 48)
        self.convN_10 = _ConvBlockUp(80, 32, 32)
        self.convN_11 = _ConvBlockUp(64, 32, 32)
        self.final = _ConvBlockOut(32, 1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(convN_3out)
        convN_5out = self.convN_5(convN_4out)
        convN_6out = self.convN_6(convN_5out)
        convN_7out = self.convN_7(torch.cat((F.interpolate(convN_6out, size=convN_5out[0, 0].shape, 
                                        mode='bilinear', align_corners= False), convN_5out), dim=1))
        convN_8out = self.convN_8(torch.cat((F.interpolate(convN_7out, size=convN_4out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_4out), dim=1))
        convN_9out = self.convN_9(torch.cat((F.interpolate(convN_8out, size=convN_3out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_3out), dim=1))
        convN_10out = self.convN_10(torch.cat((F.interpolate(convN_9out, size=convN_2out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_2out), dim=1))
        convN_11out = self.convN_11(torch.cat((F.interpolate(convN_10out, size=convN_1out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_1out), dim=1))
        final_out = self.final(convN_11out)
        return final_out

class UNet6_big(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
    - Perform 4 levels of convolution
    - When returning to the original size, concatenate output of matching sizes
    - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(UNet6_big, self).__init__()
        self.convN_1 = _ConvBlockIn(1, 40, 40)
        self.convN_2 = _ConvBlockDown(40, 40, 40)
        self.convN_3 = _ConvBlockDown(40, 40, 40)
        self.convN_4 = _ConvBlockDown(40, 60, 60)
        self.convN_5 = _ConvBlockDown(60, 60, 60)
        self.convN_6 = _ConvBlockDown(60, 60, 60)
        self.convN_7 = _ConvBlockUp(120, 60, 60)
        self.convN_8 = _ConvBlockUp(120, 60, 60)
        self.convN_9 = _ConvBlockUp(100, 60, 60)
        self.convN_10 = _ConvBlockUp(100, 40, 40)
        self.convN_11 = _ConvBlockUp(80, 40, 40)
        self.final = _ConvBlockOut(40, 1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(convN_3out)
        convN_5out = self.convN_5(convN_4out)
        convN_6out = self.convN_6(convN_5out)
        convN_7out = self.convN_7(torch.cat((F.interpolate(convN_6out, size=convN_5out[0, 0].shape, 
                                        mode='bilinear', align_corners= False), convN_5out), dim=1))
        convN_8out = self.convN_8(torch.cat((F.interpolate(convN_7out, size=convN_4out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_4out), dim=1))
        convN_9out = self.convN_9(torch.cat((F.interpolate(convN_8out, size=convN_3out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_3out), dim=1))
        convN_10out = self.convN_10(torch.cat((F.interpolate(convN_9out, size=convN_2out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_2out), dim=1))
        convN_11out = self.convN_11(torch.cat((F.interpolate(convN_10out, size=convN_1out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_1out), dim=1))
        final_out = self.final(convN_11out)
        return final_out

class MSUNet5(BaseModel):
    """
    Define the network, merge between UNet and MSNet.
    Only input when called is number of data (input) channels.
    - Perform 5 levels of convolution
    - When returning to the original size, concatenate output of matching sizes
    - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(MSUNet5, self).__init__()
        self.convN_1 = _ConvBlockIn(1, 32, 32)
        self.convN_2 = _ConvBlockDown(52, 32, 32)
        self.convN_3 = _ConvBlockDown(52, 32, 32)
        self.convN_4 = _ConvBlockDown(52, 48, 62)
        self.convN_5 = _ConvBlockDown(92, 60, 62)
        self.convN_6 = _ConvBlockUp(124, 64, 64)
        self.convN_7 = _ConvBlockUp(96, 64, 64)
        self.convN_8 = _ConvBlockUp(96, 32, 32)
        self.convN_9 = _ConvBlockUp(64, 32, 32)
        self.final = _ConvBlockOut(32, 1)
        
    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(torch.cat((x.repeat(1, 20, 1, 1), convN_1out), dim=1))
        convN_3out = self.convN_3(torch.cat((F.interpolate(x.repeat(1, 20, 1, 1), size=convN_2out[0, 0].shape,
                                        mode='bilinear', align_corners=False), convN_2out), dim=1))
        convN_4out = self.convN_4(torch.cat((F.interpolate(x.repeat(1, 20, 1, 1), size=convN_3out[0, 0].shape,
                                        mode='bilinear', align_corners=False), convN_3out), dim=1))
        convN_5out = self.convN_5(torch.cat((F.interpolate(x.repeat(1, 30, 1, 1), size=convN_4out[0, 0].shape,
                                        mode='bilinear', align_corners=False), convN_4out), dim=1))
        convN_6out = self.convN_6(torch.cat((F.interpolate(convN_5out, size=convN_4out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_4out), dim=1))
        convN_7out = self.convN_7(torch.cat((F.interpolate(convN_6out, size=convN_3out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_3out), dim=1))
        convN_8out = self.convN_8(torch.cat((F.interpolate(convN_7out, size=convN_2out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_2out), dim=1))
        convN_9out = self.convN_9(torch.cat((F.interpolate(convN_8out, size=convN_1out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_1out), dim=1))
        final_out = self.final(convN_9out)
        return final_out

class UNet5_gen(BaseModel):
    """
    Define the network. Only input when called is number of data (input) channels.
    - Perform 4 levels of convolution
    - When returning to the original size, concatenate output of matching sizes
    - The smaller domains are upsampled to the desired size with the F.upsample function.
    """
    def __init__(self, data_channels):
        super(UNet5_gen, self).__init__()
        self.convN_1 = _ConvBlockIn(1, 32, 32)
        self.convN_2 = _ConvBlockDown(32, 32, 32)
        self.convN_3 = _ConvBlockDown(32, 32, 32)
        self.convN_4 = _ConvBlockDown(32, 48, 62)
        self.convN_5 = _ConvBlockDown(62, 60, 62)
        self.convN_6 = _ConvBlockUp(124, 64, 64)
        self.convN_7 = _ConvBlockUp(96, 64, 64)
        self.convN_8 = _ConvBlockUp(96, 32, 32)
        self.convN_9 = _ConvBlockUp(64, 32, 32)
        self.final = _ConvBlockOut(32, 1)
        
    def forward(self, x):
        half_size = [int(i * 0.5) for i in list(x.size()[2:])]
        half_x = F.interpolate(x, half_size, mode='bilinear', align_corners=False)
        convN_1out = self.convN_1(half_x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(convN_3out)
        convN_5out = self.convN_5(convN_4out)
        convN_6out = self.convN_6(torch.cat((F.interpolate(convN_5out, size=convN_4out[0, 0].shape, 
                                        mode='bilinear', align_corners= False), convN_4out), dim=1))
        convN_7out = self.convN_7(torch.cat((F.interpolate(convN_6out, size=convN_3out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_3out), dim=1))
        convN_8out = self.convN_8(torch.cat((F.interpolate(convN_7out, size=convN_2out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_2out), dim=1))
        convN_9out = self.convN_9(torch.cat((F.interpolate(convN_8out, size=convN_1out[0, 0].shape, 
                                        mode='bilinear', align_corners=False), convN_1out), dim=1))
        final_out = self.final(convN_9out)
        final_out = F.interpolate(final_out, x.size()[2:], mode='bilinear', align_corners=False)
        return final_out