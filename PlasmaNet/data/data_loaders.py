########################################################################################################################
#                                                                                                                      #
#                                                     DataLoaders                                                      #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 09.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset
from ..base import BaseDataLoader


class PoissonDataLoader(BaseDataLoader):
    """
    Loads a set of charge distribution and the associated potential.
    Automatically shuffles the dataset before the validation split (see BaseDataLoader class).
    """
    def __init__(self, config, data_dir, batch_size, normalize=False, shuffle=True, validation_split=0.0, num_workers=1):
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
            self.logger.info("Using physical mormalization")
            self.data_norm = (torch.ones((physical_rhs.size(0), physical_rhs.size(1), 1, 1))) / (self.length**2)
            self.target_norm = torch.ones((potential.size(0), potential.size(1), 1, 1))
            physical_rhs /= self.data_norm
            potential /= self.target_norm
        elif self.normalize == 'empirical':
            # Value that is approximately the max of the rhs
            self.logger.info("Using empiric mormalization")
            self.data_norm = (torch.ones((physical_rhs.size(0), physical_rhs.size(1), 1, 1))) * 2e6
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

        # Create Dataset from Tensor
        self.dataset = TensorDataset(physical_rhs, potential, self.data_norm, self.target_norm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
