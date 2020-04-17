########################################################################################################################
#                                                                                                                      #
#                                                     DataLoaders                                                      #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 09.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ..base import BaseDataLoader
from ..utils import plot_dataloader_complete


class PoissonDataLoader(BaseDataLoader):
    """
    Loads a set of charge distribution and the associated potential.
    Automatically shuffles the dataset before the validation split (see BaseDataLoader class).
    """

    def __init__(self, config, data_dir, batch_size, normalize=False, shuffle=True, validation_split=0.0,
                 num_workers=1):
        self.data_dir = Path(data_dir)
        self.logger = config.get_logger('PoissonDataLoader', config['trainer']['verbosity'])

        # Load numpy files of shape (batch_size, H, W)
        physical_rhs = np.load(self.data_dir / 'physical_rhs.npy')
        potential = np.load(self.data_dir / 'potential.npy')

        # Convert to torch.Tensor of shape (batch_size, 1, H, W)
        physical_rhs = torch.from_numpy(physical_rhs[:, np.newaxis, :, :])
        potential = torch.from_numpy(potential[:, np.newaxis, :, :])

        # Normalization and length
        self.normalize = normalize
        self.length = config.length

        if self.normalize == 'max':
            self.logger.info("Using max normalization")
            self.data_norm = torch.max(torch.max(physical_rhs, 3, keepdim=True)[0], 2, keepdim=True)[0]
            self.target_norm = torch.max(torch.max(potential, 3, keepdim=True)[0], 2, keepdim=True)[0]
            physical_rhs /= self.data_norm
            potential /= self.target_norm
        elif self.normalize == 'physical':
            # For the Physical normalization we propose the following:
            # d2(pot/pot0) / d(x/L)2 = (L2 rhs0 / pot0)* rhs/rhs0
            # If mod(pot0) == 1 the normalization sums up to rhs * L**2
            # where L = physical length of the domain
            self.logger.info("Using physical normalization")
            self.data_norm = torch.ones((physical_rhs.size(0), physical_rhs.size(1), 1, 1)) / self.length**2
            self.target_norm = torch.ones((potential.size(0), potential.size(1), 1, 1))
            physical_rhs /= self.data_norm
            potential /= self.target_norm
        else:
            self.logger.info("No normalization")
            self.data_norm = torch.ones((physical_rhs.size(0), physical_rhs.size(1), 1, 1))
            self.target_norm = torch.ones((potential.size(0), potential.size(1), 1, 1))

        # Convert to torch.float32
        physical_rhs = physical_rhs.type(torch.float32)
        potential = potential.type(torch.float32)
        self.data_norm = self.data_norm.type(torch.float32)
        self.target_norm = self.target_norm.type(torch.float32)

        # if 5 channels

        if config.channels == 5:
            # Create distance tensors
            #         4
            #     - - - - -
            #    |         |
            #  1 |         | 3
            #    |         |
            #     - - - - -
            #         2
            assert (physical_rhs.size(1) == 1), "Size must be (batch_size, 1, H, W)"

            bsz  = physical_rhs.size(0)
            resX = physical_rhs.size(3)
            resY = physical_rhs.size(2)

            x_tensor = torch.arange(resX, dtype=torch.float).view((1, resX)).expand((bsz, 1, resY, resX))
            y_tensor = torch.arange(resY, dtype=torch.float).view((1, resY, 1)).expand((bsz, 1, resY, resX))

            d_1 = (potential[:, 0, :, 0].expand((bsz, 1, resY, resX)) * (resX - x_tensor) / resX).type(torch.float32)
            d_2 = (potential[:, 0, 0, :].expand((bsz, 1, resY, resX)) * (resY - y_tensor) / resY).type(torch.float32)
            d_3 = (potential[:, 0, :, -1].expand((bsz, 1, resY, resX)) * (x_tensor) / resX).type(torch.float32)
            d_4 = (potential[:, 0, -1, :].expand((bsz, 1, resY, resX)) * (y_tensor) / resY).type(torch.float32)

            # Auxiliary plot
            plot_dataloader_complete(d_1, d_2, d_3, d_4, potential, physical_rhs, x_tensor, y_tensor, config.fig_dir)

            # Final data == concatenation
            self.data = torch.cat((physical_rhs, d_1, d_2, d_3, d_4), dim=1).type(torch.float32)

        else:
            self.data = physical_rhs

        # Create Dataset from Tensor
        self.dataset = TensorDataset(self.data, potential, self.data_norm, self.target_norm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DirichletDataLoader(BaseDataLoader):
    """
    Loads a set of Dirichlet BC and the associated result to the BC problem.
    Automatically shuffles the dataset before the validation split (see BaseDataLoader class).
    """

    def __init__(self, config, data_dir, batch_size, normalize=False, shuffle=True, validation_split=0.0,
                 num_workers=1):
        self.data_dir = Path(data_dir)
        self.logger = config.get_logger('DirichletDataLoader', config['trainer']['verbosity'])

        # Load numpy file of shape (batch_size, H, W)
        potential = np.load(self.data_dir / 'potential.npy')
        BC_channel = np.zeros_like(potential)
        BC_channel[:, :, 0] = np.load(self.data_dir / 'potential_boundary.npy')

        # Convert to torch.Tensor of shape (batch_size, 1, H, W) and (batch_size, 1, 1, W) respectively
        potential = torch.from_numpy(potential[:, np.newaxis, :, :]).type(torch.float32)
        BC_channel = torch.from_numpy(BC_channel[:, np.newaxis, :, :]).type(torch.float32)
        rhs = torch.zeros_like(potential)

        # Normalization and length
        self.normalize = normalize
        self.length = config.length

        if self.normalize == 'max':
            self.logger.info("Using max normalization")
            self.data_norm = torch.max(torch.max(BC_channel, 3, keepdim=True)[0], 2, keepdim=True)[0]
            self.target_norm = torch.max(torch.max(potential, 3, keepdim=True)[0], 2, keepdim=True)[0]
            BC_channel /= self.data_norm
            potential /= self.target_norm
        elif self.normalize == 'physical':
            # For the Physical normalization we propose the following:
            # d2(pot/pot0) / d(x/L)2 = (L2 rhs0 / pot0)* rhs/rhs0
            # If mod(pot0) == 1 the normalization sums up to rhs * L**2
            # where L = physical length of the domain
            self.logger.info("Using physical normalization")
            self.data_norm = torch.ones((BC_channel.size(0), BC_channel.size(1), 1, 1)) / self.length**2
            self.target_norm = torch.ones((potential.size(0), potential.size(1), 1, 1))
            BC_channel /= self.data_norm
            potential /= self.target_norm
        else:
            self.logger.info("No normalization")
            self.data_norm = torch.ones((BC_channel.size(0), BC_channel.size(1), 1, 1))
            self.target_norm = torch.ones((potential.size(0), potential.size(1), 1, 1))

        if config.channels == 3:
            # Create distance tensor

            assert BC_channel.size(1) == 1, "Size must be (batch_size, 1, H, W)"

            bsz  = BC_channel.size(0)
            resY = BC_channel.size(2)
            resX = BC_channel.size(3)

            x_tensor = torch.arange(resX, dtype=torch.float).view((1, resX)).expand((bsz, 1, resY, resX))
            # Exponential guess of the potential from the BC data
            potential_guess = (BC_channel[:, :, :, 0].unsqueeze(3).expand((bsz, 1, resY, resX))
                               * torch.exp(-10.0 * x_tensor / resX)).type(torch.float32)

            # Final data: concatenation of rhs and BC information on respective channels
            self.data = torch.cat((rhs, BC_channel, potential_guess), dim=1).type(torch.float32)

        else:
            self.data = torch.cat((rhs, BC_channel), dim=1).type(torch.float32)

        # Convert to torch.float32
        potential = potential.type(torch.float32)
        self.data_norm = self.data_norm.type(torch.float32)
        self.target_norm = self.target_norm.type(torch.float32)

        # Create Dataset from Tensor
        self.dataset = TensorDataset(self.data, potential, self.data_norm, self.target_norm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


