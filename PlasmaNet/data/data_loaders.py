########################################################################################################################
#                                                                                                                      #
#                                                     DataLoaders                                                      #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 09.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import torch
import numpy as np
from torch.utils.data import TensorDataset
from ..base import BaseDataLoader


class PoissonDataLoader(BaseDataLoader):
    """ Loads a set of charge distribution and the associated potential. """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        physical_rhs = torch.from_numpy(np.load(data_dir / 'physical_rhs.npy'))
        potential = torch.from_numpy(np.load(data_dir / 'potential.npy'))
        self.dataset = TensorDataset(physical_rhs, potential)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
