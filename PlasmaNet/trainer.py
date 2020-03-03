########################################################################################################################
#                                                                                                                      #
#                                              Network training routines                                               #
#                                                                                                                      #
#                                Ekhi Ajuria, Guillaume Bogopolsky CERFACS, 26.02.2020                                 #
#                                                                                                                      #
########################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import no_grad
import torch.utils.data

import numpy as np
import matplotlib.pyplot


def train(epoch, model, train_loader, loss_function, optimizer, scheduler, save_value, mse_weight, gdl_weight,
          div_weight, folder):
    """ Train the model for the given epoch. """

    # Set the model to training mode
    model.train()

    # Initialize loss scores
    train_loss, train_mse, train_gdl, train_div = 0., 0., 0., 0.

    # Loop through data, sorted into batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Extract and prepare data
        if torch.cuda.is_available():  # send data to GPU
            data, target = data.cuda(), target.cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        output = model(data)

        # Compute divergence loss
        div_error = div_loss(output, data, save_value, dx, dy, epoch, batch_idx, folder)
        grad_output = grad()




