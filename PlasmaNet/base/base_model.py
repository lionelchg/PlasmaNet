########################################################################################################################
#                                                                                                                      #
#                                                   BaseModel class                                                    #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 03.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models. Overrides __str__ method and forces reimplementation of forward.
    """
    @abstractmethod
    def forward(self, *inputs):
        """ Forward pass logic """
        raise NotImplementedError

    def __str__(self):
        """ Models prints with number of trainable parameters """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
