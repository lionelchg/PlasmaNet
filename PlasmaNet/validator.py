########################################################################################################################
#                                                                                                                      #
#                                             Network validation routines                                              #
#                                                                                                                      #
#                               Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 03.03.2020                                 #
#                                                                                                                      #
########################################################################################################################

import torch
from .model.loss import laplacian_loss
from .operators.gradient import gradient_diag


def validate(epoch, model, criterion, val_loader, save_value, mse_weight, gdl_weight, div_weight, folder):
    """ Validate the model for a given epoch. """

    # Set model to eval mode
    model.eval()

    # Initialize validation scores
    val_loss, val_mse, val_gdl, val_div = 0, 0, 0, 0

    dx, dy = 5e-5  # TODO: hardcoded, to pass as argument

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Evaluate model
            output = model(data)

            # Compute loss
            div_error = laplacian_loss(output, data, save_value, dx, dy, epoch, batch_idx, folder)
            grad_output = gradient_diag(output, dx, dy)
            grad_data = gradient_diag(target, dx, dy)

            mse_loss = criterion(output, target)
            gdl_loss = criterion(grad_output, grad_data)
            div_loss = (div_error ** 2).sum()

            loss = mse_weight * mse_loss + gdl_weight * gdl_loss + div_weight * div_loss

            val_loss += loss.item()
            val_mse += (mse_weight * mse_loss).item()
            val_gdl += (gdl_weight * gdl_loss).item()
            val_div += (div_weight * div_loss).item()

    # Divide loss by dataset length
    val_loss /= len(val_loader.dataset)
    val_mse /= len(val_loader.dataset)
    val_gdl /= len(val_loader.dataset)
    val_div /= len(val_loader.dataset)

    # Print loss for the whole dataset
    print('\nTrain set: Avg loss: {:.6f}, Avg MSE : {:.6f}, Avg GDL: {:.6f}, Avg Div: {:.6f}'.format(
        val_loss, val_mse, val_gdl, val_div))

    return val_loss, val_mse, val_gdl, val_div
