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



class SingleFilter(BaseModel):
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
        super(SingleFilter, self).__init__()
        self.pad = nn.ReplicationPad2d(1)       
        self.final = nn.Conv2d(1, 1, kernel_size=3, stride=1)

    def forward(self, x, epoch):
        x = self.pad(x)
        output_fields = self.final(x)
        print("Conv Layer weight", self.final.weight)
        #output_fields = self.final(F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False))
        return output_fields
