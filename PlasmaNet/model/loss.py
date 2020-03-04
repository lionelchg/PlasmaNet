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


# Instanciate loss function
loss = MSELoss()


def laplacian_loss(field, rhs, dx, dy):
    """ A Laplacian loss function. """

    laplacian = lapl(field, dx, dy)
    lapl_loss = loss(laplacian,  - rhs)

    return lapl_loss


def electric_loss(field, target, dx, dy):
    """ Loss function on the electric field. """

    elec_output = gradient_scalar(field, dx, dy)
    elec_target = gradient_scalar(target, dx, dy)
    elec_loss = loss(elec_output, elec_target)

    return elec_loss
