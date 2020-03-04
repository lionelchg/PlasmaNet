########################################################################################################################
#                                                                                                                      #
#                                             Network validation routines                                              #
#                                                                                                                      #
#                               Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 03.03.2020                                 #
#                                                                                                                      #
########################################################################################################################

import torch
from .model.loss import laplacian_loss, electric_loss
from .operators.gradient import gradient_diag


def validate(model, criterion, val_loader, mse_weight, lapl_weight, elec_weight):
    """ Validate the model for a given epoch. """

    # Set model to eval mode
    model.eval()

    # Initialize validation scores
    val_loss, val_mse, val_lapl, val_elec = 0., 0., 0., 0.

    dx, dy = 1e-2 / 63  # TODO: hardcoded, to pass as argument

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Evaluate model
            output = model(data)

            # Compute loss
            lapl_loss = laplacian_loss(output, data, dx, dy)
            elec_loss = electric_loss(output, target, dx, dy)

            mse_loss = criterion(output, target)

            loss = mse_weight * mse_loss + lapl_weight * lapl_loss + elec_weight * elec_loss

            val_loss += loss.item()
            val_mse += (mse_weight * mse_loss).item()
            val_lapl += (lapl_weight * lapl_loss).item()
            val_elec += (elec_weight * elec_loss).item()

    # Divide loss by dataset length
    val_loss /= len(val_loader.dataset)
    val_mse /= len(val_loader.dataset)
    val_lapl /= len(val_loader.dataset)
    val_elec /= len(val_loader.dataset)

    # Print loss for the whole dataset
    print('\nVal set: Avg loss: {:.6f}, Avg MSE : {:.6f}, Avg Lapl: {:.6f}, Avg Elec: {:.6f}'.format(
        val_loss, val_mse, val_lapl, val_elec))

    return val_loss, val_mse, val_lapl, val_elec
