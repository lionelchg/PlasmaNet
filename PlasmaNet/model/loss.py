########################################################################################################################
#                                                                                                                      #
#                                                   Loss functions                                                     #
#                                                                                                                      #
#                                Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 03.03.2020                                #
#                                                                                                                      #
########################################################################################################################

from torch.nn import MSELoss
from ..operators.laplacian import laplacian as lapl
from ..operators.gradient import gradient_scalar


# Instantiate loss function
loss = MSELoss()


def laplacian_loss(field, rhs, dx, dy):
    """ A Laplacian loss function on the inside of the domain (excluding boundaries). """

    laplacian = lapl(field, dx, dy)
    lapl_loss = loss(laplacian[:, 0, 1:-1, 1:-1], - rhs[:, 0, 1:-1, 1:-1])

    return lapl_loss


def electric_loss(field, target, dx, dy):
    """ Loss function on the electric field. """

    elec_output = gradient_scalar(field, dx, dy)
    elec_target = gradient_scalar(target, dx, dy)
    elec_loss = loss(elec_output, elec_target)

    return elec_loss


def dirichlet_boundary_loss(field, target, dx, dy):
    """ Loss for Dirichlet boundaries. """

    bnd_loss = loss(field[:, 0, 0, :], target[:, 0, 0, :])
    bnd_loss += loss(field[:, 0, -1, :], target[:, 0, -1, :])
    bnd_loss += loss(field[:, 0, :, 0], target[:, 0, :, 0])
    bnd_loss += loss(field[:, 0, :, -1], target[:, 0, :, -1])

    return bnd_loss


def neumann_boundary_loss(field, target, dx, dy):
    """ Loss for Neumann boundaries. """

    # Compute electric field
    grad_field = gradient_scalar(field, dx, dy)
    grad_target = gradient_scalar(target, dx, dy)

    # Loss on the boundaries
    bnd_loss = loss(grad_field[:, 0, 1:-1, 0], grad_target[:, 0, 1:-1, 0])  # line for x = 0
    bnd_loss += loss(grad_field[:, 0, 1:-1, -1], grad_target[:, 0, 1:-1, -1])
    bnd_loss += loss(grad_field[:, 0, 0, 1:-1], grad_target[:, 0, 0, 1:-1])
    bnd_loss += loss(grad_field[:, 0, -1, 1:-1], grad_target[:, 0, -1, 1:-1])

    return bnd_loss
