########################################################################################################################
#                                                                                                                      #
#                                              Network training routines                                               #
#                                                                                                                      #
#                                Ekhi Ajuria, Guillaume Bogopolsky CERFACS, 26.02.2020                                 #
#                                                                                                                      #
########################################################################################################################

import torch
from .model.loss import laplacian_loss, electric_loss, dirichlet_boundary_loss
from .operators.gradient import gradient_diag
import matplotlib.pyplot as plt


def train(epoch, model, criterion, train_loader, optimizer, scheduler, mse_weight, lapl_weight, elec_weight,
          folder):
    """ Train the model for the given epoch. """

    # Set the model to training mode
    model.train()

    # Initialize loss scores
    train_loss, train_mse, train_lapl, train_elec = 0., 0., 0., 0.

    dx = 1e-2 / 63  # TODO: hardcoded, to pass as argument
    dy = dx

    # Loop through data, sorted into batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Extract and prepare data
        if torch.cuda.is_available():  # send data to GPU
            data, target = data.cuda(), target.cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        output = model(data)

        # Compute loss
        lapl_loss = laplacian_loss(output, data, dx, dy) + dirichlet_boundary_loss(output, data, dx, dy)
        elec_loss = electric_loss(output, target, dx, dy)

        mse_loss = criterion(output, target)

        loss = mse_weight * mse_loss + lapl_weight * lapl_loss + elec_weight * elec_loss

        # Backpropagation and optimisation
        loss.backward()
        optimizer.step()
        # scheduler.step(epoch + batch_idx / len(train_loader))

        train_loss += loss.item()
        train_mse += (mse_weight * mse_loss).item()
        train_lapl += (lapl_weight * lapl_loss).item()
        train_elec += (elec_weight * elec_loss).item()

        if epoch % 10 == 0. and batch_idx == 0:
            # Detach tensors and send them to cpu as numpy
            data_np = data.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            output_np = output.detach().cpu().numpy()

            # Lots of plots
            fig, axes = plt.subplots(figsize=(12, 25), nrows=5, ncols=3)
            fig.suptitle(' Model {} for epoch {}'.format(batch_idx, epoch))

            for k in range(5):

                tt = axes[k, 0].imshow(data_np[batch_idx + k, 0], origin='lower')
                axes[k, 0].set_title('rhs')
                axes[k, 0].axis('off')
                fig.colorbar(tt, ax=axes[k, 0])

                tt = axes[k, 1].imshow(output_np[batch_idx + k, 0], origin='lower')
                axes[k, 1].set_title('predicted potential')
                axes[k, 1].axis('off')
                fig.colorbar(tt, ax=axes[k, 1])

                tt = axes[k, 2].imshow(target_np[batch_idx + k, 0], origin='lower')
                axes[k, 2].set_title('target potential')
                axes[k, 2].axis('off')
                fig.colorbar(tt, ax=axes[k, 2])

            plt.tight_layout()
            plt.savefig(folder + '/Train_Images' + '/Model_{}_Result_{:06d}.png'.format(batch_idx, epoch))
            plt.close('all')

    # Divide loss by dataset length
    train_loss /= len(train_loader.dataset)
    train_mse /= len(train_loader.dataset)
    train_lapl /= len(train_loader.dataset)
    train_elec /= len(train_loader.dataset)

    # Print loss for the whole dataset
    print('\nTrain set: Avg loss: {:.6f}, Avg MSE : {:.6f}, Avg Lapl: {:.6f}, Avg Elec: {:.6f}'.format(
        train_loss, train_mse, train_lapl, train_elec))

    return train_loss, train_mse, train_lapl, train_elec
