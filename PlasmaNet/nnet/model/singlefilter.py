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
    Define the network. The input needed are the number of data (input) channels.
    And the kernel size (to be added in the corresponding config file)

    """
    def __init__(self, data_channels, kernel):
        super(SingleFilter, self).__init__()
        self.pad = nn.ReplicationPad2d(int((kernel-1)/2))       
        self.final = nn.Conv2d(1, 1, kernel_size=kernel, stride=1)

    def forward(self, x, epoch):
        x = self.pad(x)
        output_fields = self.final(x)
        print("Conv Layer weight", self.final.weight)

        return output_fields
