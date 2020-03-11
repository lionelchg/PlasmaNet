########################################################################################################################
#                                                                                                                      #
#                                                  Base Loss class                                                     #
#                                                                                                                      #
#                                      Guillaume Bogopolsky, CERFACS, 11.03.2020                                       #
#                                                                                                                      #
########################################################################################################################

from torch.nn.modules import _Loss
from abc import abstractmethod


class BaseLoss(_Loss):
    """
    Base class for all losses, forces definition of input checking methods.
    `require_input_data` returns whether the data (rhs here) is needed for the forward step (e.g. for laplacian loss).
    By default, it is not needed.
    """
    def __init__(self):
        super().__init__()
        self._require_input_data = False

    @abstractmethod
    def forward(self, output, target, **kwargs):
        """ Computes the loss (implemented as __call__ in nn.Module, from which it inherits). """
        raise NotImplementedError

    def require_input_data(self):
        """ Check whether forward method needs the data (here rhs) for computation (e.g. for laplacian loss). """
        return self._require_input_data
