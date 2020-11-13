########################################################################################################################
#                                                                                                                      #
#                                   Single filter network for testing purposes                                         #
#                                                                                                                      #
#                                          Ekhi Ajuria, CERFACS, 13.10.2020                                            #
#                                                                                                                      #
########################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel



class SingleFilter(BaseModel):
    """
    Define the network. The only input needed is the number of data (input) channels.

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
