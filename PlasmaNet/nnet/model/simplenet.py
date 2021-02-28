########################################################################################################################
#                                                                                                                      #
#                                          Simple network for testing purposes                                         #
#                                                                                                                      #
#                                          Ekhi Ajuria, CERFACS, 13.10.2020                                            #
#                                                                                                                      #
########################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel



class SimpleNet(BaseModel):
    """
    Define the network. The only input needed is the number of data (input) channels.
    And the number of filters.
    The network will only two conv2D layers with an intermediate activation function.
    The order is: Pad Conv ReLu Pad Conv

    """
    def __init__(self, data_channels, filters):
        super(SimpleNet, self).__init__()
        self.pad = nn.ReplicationPad2d(1)      
        self.initial = nn.Conv2d(1, filters, kernel_size=3, stride=1)
        self.final =  nn.Conv2d(filters, 1, kernel_size=3, stride=1)

    def forward(self, x, epoch):
        x = self.pad(x)
        x = self.initial(x)
        x = F.relu(x)
        x = self.pad(x)
        x = self.final(x)

        return x
