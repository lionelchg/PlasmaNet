########################################################################################################################
#                                                                                                                      #
#                                             Network validation routines                                              #
#                                                                                                                      #
#                               Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 03.03.2020                                 #
#                                                                                                                      #
########################################################################################################################

import torch
from .model.loss import laplacian_loss, electric_loss, dirichlet_boundary_loss
from .operators.gradient import gradient_diag
import matplotlib.pyplot as plt


def validate(epoch, model, criterion, val_loader, mse_weight, lapl_weight, elec_weight, folder):
    """ Validate the model for a given epoch. """

    # Set model to eval mode
    model.eval()

    # Initialize validation scores
    val_loss, val_mse, val_lapl, val_elec = 0., 0., 0., 0.

    dx = 1e-2 / 63  # TODO: hardcoded, to pass as argument
    dy = dx

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Evaluate model
            output = model(data)

            # Compute loss
            lapl_loss = laplacian_loss(output, data, dx, dy) + dirichlet_boundary_loss(output, data, dx, dy)
            elec_loss = electric_loss(output, target, dx, dy)

            mse_loss = criterion(output, target)

            loss = mse_weight * mse_loss + lapl_weight * lapl_loss + elec_weight * elec_loss

            val_loss += loss.item()
            val_mse += (mse_weight * mse_loss).item()
            val_lapl += (lapl_weight * lapl_loss).item()
            val_elec += (elec_weight * elec_loss).item()

            if epoch % 10 == 0. and batch_idx == 0:
                fig, axes = plt.subplots(figsize=(5, 12), nrows=3, ncols=1)
                ax1, ax2, ax3 = axes.ravel()
                fig.suptitle(' Model {} for epoch {}'.format(batch_idx, epoch))

                tt = ax1.imshow(data[batch_idx, 0].detach().cpu().numpy(), origin='lower')
                ax1.set_title('rhs')
                ax1.axis('off')
                fig.colorbar(tt, ax=ax1)

                tt = ax2.imshow(output[batch_idx, 0].detach().cpu().numpy(), origin='lower')
                ax2.set_title('predicted potential')
                ax2.axis('off')
                fig.colorbar(tt, ax=ax2)

                tt = ax3.imshow(target[batch_idx, 0].detach().cpu().numpy(), origin='lower')
                ax3.set_title('target potential')
                ax3.axis('off')
                fig.colorbar(tt, ax=ax3)

                plt.tight_layout()
                plt.savefig(folder + '/Val_Images' + '/Model_{}_Result_{:06d}.png'.format(batch_idx, epoch))
                plt.close('all')

    # Divide loss by dataset length
    val_loss /= len(val_loader.dataset)
    val_mse /= len(val_loader.dataset)
    val_lapl /= len(val_loader.dataset)
    val_elec /= len(val_loader.dataset)

    # Print loss for the whole dataset
    print('\nVal set: Avg loss: {:.6f}, Avg MSE : {:.6f}, Avg Lapl: {:.6f}, Avg Elec: {:.6f}'.format(
        val_loss, val_mse, val_lapl, val_elec))

    return val_loss, val_mse, val_lapl, val_elec
