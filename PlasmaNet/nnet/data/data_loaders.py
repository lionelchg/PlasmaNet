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
import os

from ..base import BaseDataLoader
from ..utils import plot_dataloader_complete, fourier_guess, dataset_input_filter


class PoissonDataLoader(BaseDataLoader):
    """
    Loads a set of charge distribution and the associated potential.
    Automatically shuffles the dataset before the validation split (see BaseDataLoader class).
    """

    def __init__(self, config, data_dir, batch_size, normalize=False, shuffle=True, validation_split=0.0,
                 input_cutoff_frequency=None, num_workers=1, scaling_factor=1.0, alpha=0.1):
        self.data_dir = Path(data_dir)
        self.logger = config.get_logger('PoissonDataLoader', config['globals']['verbosity'])

        # Load numpy files of shape (batch_size, H, W)
        physical_rhs = np.load(self.data_dir / 'physical_rhs.npy')
        potential = np.load(self.data_dir / 'potential.npy')

        # Filter input potential if asked for
        if input_cutoff_frequency is not None and input_cutoff_frequency != 'None':
            dataset_input_filter(physical_rhs, input_cutoff_frequency, config.size, config.length)
        self.multi = os.path.exists(self.data_dir / 'potential_interp.npy')

        if self.multi:
            potential_multi = np.load(self.data_dir / 'potential_interp.npy')

        # Convert to torch.Tensor of shape (batch_size, 1, H, W)
        physical_rhs = torch.from_numpy(physical_rhs[:, np.newaxis, :, :])
        potential = torch.from_numpy(potential[:, np.newaxis, :, :])
        if self.multi:
            potential_multi = torch.from_numpy(potential_multi)

        # Normalization and length
        self.normalize = normalize
        self.Lx = config.Lx
        self.Ly = config.Ly

        if self.normalize == 'max':
            self.logger.info("Using max normalization")
            self.data_norm = torch.max(torch.max(physical_rhs, 3, keepdim=True)[0], 2, keepdim=True)[0]
            self.target_norm = torch.max(torch.max(potential, 3, keepdim=True)[0], 2, keepdim=True)[0]
            physical_rhs /= self.data_norm
            potential /= self.target_norm
        elif self.normalize == 'analytical' or self.normalize == 'empirical':
            self.logger.info("Using analytical normalization from Fourier series solution")
            self.alpha = alpha
            self.ratio_max = ratio_potrhs(self.alpha, self.Lx, self.Ly)
            self.data_norm = torch.ones((physical_rhs.size(0), physical_rhs.size(1), 1, 1)) / self.ratio_max
            self.target_norm = torch.ones((potential.size(0), potential.size(1), 1, 1))
            if self.multi:
                self.target_norm_m = torch.ones((potential_multi.size(0), potential_multi.size(1), 1, 1))
            physical_rhs /= self.data_norm
        else:
            self.logger.info("No normalization")
            self.data_norm = torch.ones((physical_rhs.size(0), physical_rhs.size(1), 1, 1))
            self.target_norm = torch.ones((potential.size(0), potential.size(1), 1, 1))

        # Scaling factor for the float32 that are not very precise
        self.scaling_factor = scaling_factor
        physical_rhs *= self.scaling_factor
        potential *= self.scaling_factor

        # Convert to torch.float32
        physical_rhs = physical_rhs.type(torch.float32)
        potential = potential.type(torch.float32)
        self.data_norm = self.data_norm.type(torch.float32)
        self.target_norm = self.target_norm.type(torch.float32)
        if self.multi:
            self.target_norm_m = self.target_norm_m.type(torch.float32)
            potential_multi = potential_multi.type(torch.float32)
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

            d_1 = (potential[:, :, :, 0].unsqueeze(3).expand((bsz, 1, resY, resX)) * (resX - x_tensor) / resX).type(torch.float32)
            d_2 = (potential[:, :, 0, :].unsqueeze(2).expand((bsz, 1, resY, resX)) * (resY - y_tensor) / resY).type(torch.float32)
            d_3 = (potential[:, :, :, -1].unsqueeze(3).expand((bsz, 1, resY, resX)) * (x_tensor) / resX).type(torch.float32)
            d_4 = (potential[:, :, -1, :].unsqueeze(2).expand((bsz, 1, resY, resX)) * (y_tensor) / resY).type(torch.float32)

            # Auxiliary plot
            plot_dataloader_complete(d_1, d_2, d_3, d_4, potential, physical_rhs, x_tensor, y_tensor, config.fig_dir)

            # Final data == concatenation
            self.data = torch.cat((physical_rhs, d_1, d_2, d_3, d_4), dim=1).type(torch.float32)

        else:
            self.data = physical_rhs

        # Create Dataset from Tensor
        if self.multi:
            self.dataset = TensorDataset(self.data, potential_multi, self.data_norm, self.target_norm)
        else:
            self.dataset = TensorDataset(self.data, potential, self.data_norm, self.target_norm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

def ratio_potrhs(alpha, Lx, Ly):
    """ Potential over RHS ratio for 2D dirichlet problem

    :param alpha: The normalization coefficient set by the user
    :type alpha: float
    :param Lx: Length of the domain in the x direction
    :type Lx: float
    :param Ly: Length of the domain in the y direction
    :type Ly: float
    :return: Ratio of potential over RHS
    :rtype: float
    """
    return alpha / (np.pi**2 / 4)**2 / (1 / Lx**2 + 1 / Ly**2)

class PhotoDataLoader(BaseDataLoader):
    """
    Loads a set of charge distribution and the associated potential.
    Automatically shuffles the dataset before the validation split (see BaseDataLoader class).
    """

    def __init__(self, config, data_dir, batch_size, sph_file, lambda_scale=1.0, shuffle=True, validation_split=0.0,
                num_workers=1, scaling_factor=1.0, normalize=1.0):
        self.data_dir = Path(data_dir)
        self.logger = config.get_logger('PoissonDataLoader', config['globals']['verbosity'])

        # Load numpy files of shape (batch_size, H, W)
        ioniz_rate = np.load(self.data_dir / 'ioniz_rate.npy')
        # Sph = np.load(self.data_dir / 'Sph.npy')
        Sph = np.load(self.data_dir / sph_file)

        # Convert to torch.Tensor of shape (batch_size, 1, H, W)
        ioniz_rate = torch.from_numpy(ioniz_rate[:, np.newaxis, :, :])
        Sph = torch.from_numpy(Sph[:, np.newaxis, :, :])

        # Normalization and length
        self.Lx = config.Lx
        self.Ly = config.Ly

        # Normalization
        self.normalize = normalize
        self.data_norm = torch.ones((ioniz_rate.size(0), ioniz_rate.size(1), 1, 1)) * self.normalize
        self.target_norm = torch.ones((Sph.size(0), Sph.size(1), 1, 1))
        ioniz_rate /= self.data_norm

        # Scaling factor for the float32 that are not very precise
        self.scaling_factor = scaling_factor
        ioniz_rate *= self.scaling_factor
        Sph *= self.scaling_factor

        # Convert to torch.float32
        ioniz_rate = ioniz_rate.type(torch.float32)
        Sph = Sph.type(torch.float32)
        self.data_norm = self.data_norm.type(torch.float32)
        self.target_norm = self.target_norm.type(torch.float32)

        self.data = ioniz_rate

        # Create Dataset from Tensor
        self.dataset = TensorDataset(self.data, Sph, self.data_norm, self.target_norm)

        # Add lambda scaling factor to better match the desired photoionization values
        self.lambda_scale= lambda_scale

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)