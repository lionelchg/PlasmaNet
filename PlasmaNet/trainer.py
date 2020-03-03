########################################################################################################################
#                                                                                                                      #
#                                              Network training routines                                               #
#                                                                                                                      #
#                                Ekhi Ajuria, Guillaume Bogopolsky CERFACS, 26.02.2020                                 #
#                                                                                                                      #
########################################################################################################################

import torch
from .model.loss import laplacian_loss
from .operators.gradient import gradient_diag


def train(epoch, model, criterion, train_loader, optimizer, scheduler, save_value, mse_weight, gdl_weight,
          div_weight, folder):
    """ Train the model for the given epoch. """

    # Set the model to training mode
    model.train()

    # Initialize loss scores
    train_loss, train_mse, train_gdl, train_div = 0., 0., 0., 0.

    dx, dy = 5e-5  # TODO: hardcoded, to pass as argument

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
        div_error = laplacian_loss(output, data, save_value, dx, dy, epoch, batch_idx, folder)
        grad_output = gradient_diag(output, dx, dy)
        grad_data = gradient_diag(target, dx, dy)

        mse_loss = criterion(output, target)
        gdl_loss = criterion(grad_output, grad_data)
        div_loss = (div_error**2).sum()

        loss = mse_weight * mse_loss + gdl_weight * gdl_loss + div_weight * div_loss

        # Backpropagation and optimisation
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + batch_idx / len(train_loader))

        train_loss += loss.item()
        train_mse += (mse_weight * mse_loss).item()
        train_gdl += (gdl_weight * gdl_loss).item()
        train_div += (div_weight * div_loss).item()

    # Divide loss by dataset length
    train_loss /= len(train_loader.dataset)
    train_mse /= len(train_loader.dataset)
    train_gdl /= len(train_loader.dataset)
    train_div /= len(train_loader.dataset)

    # Print loss for the whole dataset
    print('\nTrain set: Avg loss: {:.6f}, Avg MSE : {:.6f}, Avg GDL: {:.6f}, Avg Div: {:.6f}'.format(
        train_loss, train_mse, train_gdl, train_div))

    return train_loss, train_mse, train_gdl, train_div
