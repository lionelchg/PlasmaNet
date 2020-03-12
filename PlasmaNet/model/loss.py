########################################################################################################################
#                                                                                                                      #
#                                                    Loss classes                                                      #
#                                                                                                                      #
#                                Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 03.03.2020                                #
#                                                                                                                      #
########################################################################################################################

from ..base.base_loss import BaseLoss
import torch.nn.functional as F
from ..operators.laplacian import laplacian as lapl
from ..operators.gradient import gradient_scalar


class InsideLoss(BaseLoss):
    """ Computes the weighted MSELoss of the interior of the domain (excluding boundaries). """
    def __init__(self, inside_weight):
        super().__init__()
        self.weight = inside_weight

    def forward(self, output, target, **kwargs):
        return F.mse_loss(output[:, 0, 1:-1, 1:-1], target[:, 0, 1:-1, 1:-1]) * self.weight


class LaplacianLoss(BaseLoss):
    """ A Laplacian loss function on the inside of the domain (excluding boundaries). """
    def __init__(self, lapl_weight, dx, dy):
        super().__init__()
        self.weight = lapl_weight
        self.dx = dx
        self.dy = dy
        if isinstance(self.dx, str) or isinstance(self.dy, str):
            self.dx = eval(dx)
            self.dy = eval(dy)
        self._require_input_data = True  # Need rhs for computation

    def forward(self, output, target, data=None, **kwargs):
        laplacian = lapl(output, self.dx, self.dy)
        return F.mse_loss(laplacian[:, 0, 1:-1, 1:-1], - data[:, 0, 1:-1, 1:-1]) * self.weight


class ElectricLoss(BaseLoss):
    """ Loss function on the electric field. """
    def __init__(self, elec_weight, dx, dy):
        super().__init__()
        self.weight = elec_weight
        self.dx = dx
        self.dy = dy
        if isinstance(self.dx, str) or isinstance(self.dy, str):
            self.dx = eval(dx)
            self.dy = eval(dy)
        self._require_input_data = False

    def forward(self, output, target, **kwargs):
        elec_output = gradient_scalar(output, self.dx, self.dy)
        elec_target = gradient_scalar(target, self.dx, self.dy)
        return F.mse_loss(elec_output, elec_target) * self.weight


class DirichletBoundaryLoss(BaseLoss):
    """ Loss for Dirichlet boundaries. """
    def __init__(self, bound_weight):
        super().__init__()
        self.weight = bound_weight

    def forward(self, output, target, **kwargs):
        bnd_loss = F.mse_loss(output[:, 0, 0, :], target[:, 0, 0, :])
        bnd_loss += F.mse_loss(output[:, 0, -1, :], target[:, 0, -1, :])
        bnd_loss += F.mse_loss(output[:, 0, :, 0], target[:, 0, :, 0])
        bnd_loss += F.mse_loss(output[:, 0, :, -1], target[:, 0, :, -1])
        return bnd_loss * self.weight


class NeumannBoundaryLoss(BaseLoss):
    """ Loss for Neumann boundaries. """
    def __init__(self, bound_weight, dx, dy):
        super().__init__()
        self.weight = bound_weight
        self.dx = dx
        self.dy = dy
        if isinstance(self.dx, str) or isinstance(self.dy, str):
            self.dx = eval(dx)
            self.dy = eval(dy)

    def forward(self, output, target, **kwargs):
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
    Class for loss composition.
    TODO: pass list of classes as argument? With list of object, to avoid manually listing losses
    """
    def __init__(self, inside_weight, bound_weight, elec_weight, lapl_weight, dx, dy):
        super().__init__()
        self.inside_loss = InsideLoss(inside_weight)
        self.bound_loss = DirichletBoundaryLoss(bound_weight)
        self.elec_loss = ElectricLoss(elec_weight, dx, dy)
        self.lapl_loss = LaplacianLoss(lapl_weight, dx, dy)
        self._require_input_data = True

    def forward(self, output, target, data=None, **kwargs):
        composed_loss = self.inside_loss.forward(output, target)
        composed_loss += self.bound_loss.forward(output, target)
        composed_loss += self.elec_loss.forward(output, target)
        composed_loss += self.lapl_loss.forward(output, target, data)
        return composed_loss
