########################################################################################################################
#                                                                                                                      #
#                                                    Loss classes                                                      #
#                                                                                                                      #
#                                Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 03.03.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch
from ..base.base_loss import BaseLoss
import torch.nn.functional as F
from ..operators.laplacian import laplacian as lapl
from ..operators.gradient import gradient_scalar
import sys


class InsideLoss(BaseLoss):
    """ Computes the weighted MSELoss of the interior of the domain (excluding boundaries). """
    def __init__(self, inside_weight, **_):
        super().__init__()
        self.weight = inside_weight

    def _forward(self, output, target, **kwargs):
        return F.mse_loss(output[:, 0, 1:-1, 1:-1], target[:, 0, 1:-1, 1:-1]) * self.weight


class LaplacianLoss(BaseLoss):
    """ A Laplacian loss function on the inside of the domain (excluding boundaries). """
    def __init__(self, lapl_weight, dx, dy, **_):
        super().__init__()
        self.weight = lapl_weight
        self.dx = dx
        self.dy = dy
        if isinstance(self.dx, str) or isinstance(self.dy, str):
            self.dx = eval(dx)
            self.dy = eval(dy)
        self._require_input_data = True  # Need rhs for computation

    def _forward(self, output, target, data=None, target_norm=1., data_norm=1., **_):
        laplacian = lapl(output * target_norm / data_norm, self.dx, self.dy)
        return F.mse_loss(laplacian[:, 0, 1:-1, 1:-1], - data[:, 0, 1:-1, 1:-1]) * self.weight


class ElectricLoss(BaseLoss):
    """ Loss function on the electric field. """
    def __init__(self, elec_weight, dx, dy, **_):
        super().__init__()
        self.weight = elec_weight
        self.dx = dx
        self.dy = dy
        if isinstance(self.dx, str) or isinstance(self.dy, str):
            self.dx = eval(dx)
            self.dy = eval(dy)
        self._require_input_data = False

    def _forward(self, output, target, **_):
        elec_output = gradient_scalar(output, self.dx, self.dy)
        elec_target = gradient_scalar(target, self.dx, self.dy)
        return F.mse_loss(elec_output, elec_target) * self.weight


class DirichletBoundaryLoss(BaseLoss):
    """ Loss for Dirichlet boundaries. """
    def __init__(self, bound_weight, **_):
        super().__init__()
        self.weight = bound_weight

    def _forward(self, output, target, **_):
        bnd_loss = F.mse_loss(output[:, 0, 0, :], target[:, 0, 0, :])
        bnd_loss += F.mse_loss(output[:, 0, -1, :], target[:, 0, -1, :])
        bnd_loss += F.mse_loss(output[:, 0, :, 0], target[:, 0, :, 0])
        bnd_loss += F.mse_loss(output[:, 0, :, -1], target[:, 0, :, -1])
        return bnd_loss * self.weight


class NeumannBoundaryLoss(BaseLoss):
    """ Loss for Neumann boundaries. """
    def __init__(self, bound_weight, dx, dy, **_):
        super().__init__()
        self.weight = bound_weight
        self.dx = dx
        self.dy = dy
        if isinstance(self.dx, str) or isinstance(self.dy, str):
            self.dx = eval(dx)
            self.dy = eval(dy)

    def _forward(self, output, target, **_):
        # Compute electric field
        grad_output = gradient_scalar(output, self.dx, self.dy)
        grad_target = gradient_scalar(target, self.dx, self.dy)
        # Loss on the boundaries
        bnd_loss = F.mse_loss(grad_output[:, 0, 1:-1, 0], grad_target[:, 0, 1:-1, 0])  # line for x = 0
        bnd_loss += F.mse_loss(grad_output[:, 0, 1:-1, -1], grad_target[:, 0, 1:-1, -1])
        bnd_loss += F.mse_loss(grad_output[:, 0, 0, 1:-1], grad_target[:, 0, 0, 1:-1])
        bnd_loss += F.mse_loss(grad_output[:, 0, -1, 1:-1], grad_target[:, 0, -1, 1:-1])
        return bnd_loss * self.weight


class ComposedLoss(BaseLoss):
    """
    Class for loss composition, with list of losses to compose as argument.
    """
    def __init__(self, loss_list, **kwargs):
        """
        Initializes the ComposedLoss with the loss specified in loss_list, and pass the arguments by dictionary
        from the input file.
        Therefore, the loss must have different argument name for the weight for example.
        """
        super().__init__()
        # Initializes all losses in a list
        self.loss_list = loss_list
        self.losses = list()
        for loss in self.loss_list:
            try:
                self.losses.append(getattr(sys.modules[__name__], loss)(**kwargs))  # Look for classes in this module
            except TypeError as err:
                print('while initializing {} class:'.format(loss))
                raise type(err)('{} while initializing {} class.'.format(err, loss))
        # Check if any loss requires data input
        self._require_input_data = any([loss.require_input_data() for loss in self.losses])
        # Store results for logger access
        self.results = torch.zeros(len(self.loss_list))  # Use numpy to ensure floating-point sum

    def _forward(self, output, target, **kwargs):
        out = self.losses[0](output, target, **kwargs)
        with torch.no_grad():
            self.results[0] = self.losses[0](output, target, **kwargs)
        for i, loss in enumerate(self.losses[1:]):
            out += loss(output, target, **kwargs)
            with torch.no_grad():
                self.results[i] = loss(output, target, **kwargs)
        return out

    def log(self):
        """ Returns log of each loss independently as a dict() with loss names keys and the associated torch values. """
        return {loss_name: result for loss_name, result in zip(self.loss_list, self.results)}
