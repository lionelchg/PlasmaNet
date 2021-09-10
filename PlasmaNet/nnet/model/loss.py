########################################################################################################################
#                                                                                                                      #
#                                                    Loss classes                                                      #
#                                                                                                                      #
#                                Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 03.03.2020                                #
#                                                                                                                      #
########################################################################################################################

import sys

import scipy.constants as co
import torch
import torch.nn.functional as F

from ..base.base_loss import BaseLoss
from ..operators.gradient import gradient_scalar
from ..operators.laplacian import laplacian as lapl


class InsideLoss(BaseLoss):
    """ Computes the weighted MSELoss of the interior of the domain (excluding boundaries). """
    def __init__(self, config, inside_weight, **_):
        super().__init__()
        self.weight = inside_weight
        self.cyl = config.coord == 'cyl'

    def _forward(self, output, target, **kwargs):
        if self.cyl:
            return F.mse_loss(output[:, 0, 0:-1, 1:-1], target[:, 0, 0:-1, 1:-1]) * self.weight
        else:
            return F.mse_loss(output[:, 0, 1:-1, 1:-1], target[:, 0, 1:-1, 1:-1]) * self.weight

class InsideAxialLoss(BaseLoss):
    """ Computes the weighted MSELoss of the BC in the axis (for the cylindrical case). """
    def __init__(self, config, axial_weight, **_):
        super().__init__()
        self.weight = axial_weight

    def _forward(self, output, target, **kwargs):
        return F.mse_loss(output[:, 0, 0, 1:-1], target[:, 0, 0, 1:-1]) * self.weight


class MSInsideLoss_n(BaseLoss):
    """ Computes the weighted MSELoss of the interior of the domain (excluding boundaries). For the n scale"""
    def __init__(self, config, inside_weight_n, **_):
        super().__init__()
        self.weight_n = inside_weight_n
    def _forward(self, output, target, **kwargs):
        return F.mse_loss(output[:, 1, 1:-1, 1:-1], target[:, 0, 1:-1, 1:-1]) * self.weight_n

class MSInsideLoss_n2(BaseLoss):
    """ Computes the weighted MSELoss of the interior of the domain (excluding boundaries). For the n2 scale"""
    def __init__(self, config, inside_weight_n2, **_):
        super().__init__()
        self.weight_n2 = inside_weight_n2
    def _forward(self, output, target, **kwargs):
        return F.mse_loss(output[:, 2, 1:-1, 1:-1], target[:, 1, 1:-1, 1:-1]-target[:, 2, 1:-1, 1:-1]) * self.weight_n2

class MSInsideLoss_n4(BaseLoss):
    """ Computes the weighted MSELoss of the interior of the domain (excluding boundaries). For the n4 scale"""
    def __init__(self, config, inside_weight_n4, **_):
        super().__init__()
        self.weight_n4 = inside_weight_n4
    def _forward(self, output, target, **kwargs):
        return F.mse_loss(output[:, 3, 1:-1, 1:-1], target[:, 2, 1:-1, 1:-1]) * self.weight_n4


class LaplacianLoss(BaseLoss):
    """ A Laplacian loss function on the inside of the domain (excluding boundaries). """
    def __init__(self, config, lapl_weight, **_):
        super().__init__()
        self.weight = lapl_weight
        self.dx = config.dx
        self.dy = config.dy
        self.Lx = config.Lx
        self.r_nodes = config.r_nodes
        if self.r_nodes is not None:  # in this case, a NumPy array
            self.r_nodes = torch.from_numpy(self.r_nodes).cuda()
        self._require_input_data = True  # Need rhs for computation
        self.cyl = config.coord == 'cyl'

    def _forward(self, output, target, data=None, target_norm=1., data_norm=1., **_):
        laplacian = lapl(output * target_norm / data_norm, self.dx, self.dy, r=self.r_nodes)
        if self.cyl:
            return self.Lx**4 * F.mse_loss(laplacian[:, 0, 0:-1, 1:-1], - data[:, 0, 0:-1, 1:-1]) * self.weight
        else:
            return self.Lx**4 * F.mse_loss(laplacian[:, 0, 1:-1, 1:-1], - data[:, 0, 1:-1, 1:-1]) * self.weight


class EnergyLoss(BaseLoss):
    """ An Energy loss that is minimum for Poisson's equation (Variational approach). """
    def __init__(self, config, energy_weight, **_):
        super().__init__()
        self.weight = energy_weight
        self.dx = config.dx
        self.dy = config.dy
        self.n_inputs = config.nnx * config.nny * config.batch_size
        self._require_input_data = True  # Need rhs for computation

    def _forward(self, output, target, data=None, target_norm=1., data_norm=1., **_):
        elec_output = gradient_scalar(output, self.dx, self.dy)
        # Need to use data_norm because the rhs has been changed due to scaling
        norm_elec_output = (elec_output[:, 0, :, :] ** 2 + elec_output[:, 1, :, :] ** 2).unsqueeze(1) / data_norm
        energy_output = (0.5 * norm_elec_output - output * data)
        return co.epsilon_0 * torch.sum(energy_output) / self.n_inputs * self.weight


class ElectricLoss(BaseLoss):
    """ Loss function on the electric field. """
    def __init__(self, config, elec_weight, **_):
        super().__init__()
        self.weight = elec_weight
        self.dx = config.dx
        self.dy = config.dy
        self._require_input_data = False

    def _forward(self, output, target, target_norm=1., data_norm=1., **_):
        elec_output = gradient_scalar(output * target_norm / data_norm, self.dx, self.dy)
        elec_target = gradient_scalar(target * target_norm / data_norm, self.dx, self.dy)
        return F.mse_loss(elec_output, elec_target) * self.weight


class DirichletBoundaryLoss(BaseLoss):
    """ Loss for Dirichlet boundaries. self.cyl boolean to determine if the y = 0 boundary is dirichlet or not """
    def __init__(self, config, bound_weight, **_):
        super().__init__()
        self.weight = bound_weight
        self.cyl = config.coord == 'cyl'

    def _forward(self, output, target, **_):
        bnd_loss = F.mse_loss(output[:, 0, -1, :], torch.zeros_like(output[:, 0, -1, :]))
        bnd_loss += F.mse_loss(output[:, 0, :, 0], torch.zeros_like(output[:, 0, :, 0]))
        bnd_loss += F.mse_loss(output[:, 0, :, -1], torch.zeros_like(output[:, 0, :, -1]))
        # Axis points to add only if we're in cartesian coordinates
        if self.cyl: bnd_loss += F.mse_loss(output[:, 0, 0, :], torch.zeros_like(output[:, 0, 0, :]))
        return bnd_loss * self.weight


class NeumannBoundaryLoss(BaseLoss):
    """ Loss for Neumann boundaries. """
    def __init__(self, config, bound_weight, **_):
        super().__init__()
        self.weight = bound_weight
        self.dx = config.dx
        self.dy = config.dy

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


class LongTermLaplacianLoss(LaplacianLoss):
    """ A Laplacian loss function on the inside of the domain (excluding boundaries),
    for a long term loss. """
    def __init__(self, config, lt_weight, **_):
        super().__init__(config, lt_weight)
    def _forward(self, output, target, data=None, target_norm=1., data_norm=1., **_):
        laplacian = lapl(output[:,1].unsqueeze(1) * target_norm / data_norm, self.dx, self.dy, r=self.r_nodes)
        return F.mse_loss(laplacian[:, 0, 1:-1, 1:-1], - data[:, 1, 1:-1, 1:-1]) * self.weight
        

class ComposedLoss(BaseLoss):
    """
    Class for loss composition, with list of losses to compose as argument.
    """
    def __init__(self, config, loss_list, **kwargs):
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
                self.losses.append(getattr(sys.modules[__name__], loss)(config, **kwargs))  # Look for classes in this module
            except TypeError as err:
                print('while initializing {} class:'.format(loss))
                raise type(err)('{} while initializing {} class.'.format(err, loss))
        # Check if any loss requires data input
        self._require_input_data = any([loss.require_input_data() for loss in self.losses])
        # Store results for logger access
        self.results = torch.zeros(len(self.loss_list))  # Use numpy to ensure floating-point sum

    def _forward(self, output, target, **kwargs):
        out = self.losses[0](output, target, **kwargs)
        self.results[0].copy_(out)
        for i, loss in enumerate(self.losses[1:]):
            tmp = loss(output, target, **kwargs)
            out += tmp
            self.results[i + 1].copy_(tmp)
        return out

    def log(self):
        """ Returns log of each loss independently as a dict() with loss names keys and the associated torch values. """
        out = {loss_name: result for loss_name, result in zip(self.loss_list, self.results)}
        out.update({'loss': self.result})
        return out
