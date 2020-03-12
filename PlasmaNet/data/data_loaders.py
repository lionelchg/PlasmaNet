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
    def __init__(self, data_dir, batch_size, normalize=False, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = Path(data_dir)

        # Load numpy files of shape (batch_size, H, W)
        physical_rhs = np.load(self.data_dir / 'physical_rhs.npy')
        potential = np.load(self.data_dir / 'potential.npy')

        # Normalization
        self.normalize = normalize
        if self.normalize:
            self.data_norm = physical_rhs.max()
            self.target_norm = potential.max()
            physical_rhs /= self.data_norm
            potential /= self.target_norm

        # Convert to torch.Tensor of shape (batch_size, 1, H, W)
        physical_rhs = torch.from_numpy(physical_rhs[:, np.newaxis, :, :]).type(torch.float32)
        potential = torch.from_numpy(potential[:, np.newaxis, :, :]).type(torch.float32)

        # Create Dataset from Tensor
        self.dataset = TensorDataset(physical_rhs, potential)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
