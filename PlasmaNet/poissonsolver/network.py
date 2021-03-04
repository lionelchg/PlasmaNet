import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.sparse.linalg import spsolve

import torch
import argparse
import collections

from .base import BasePoisson
import PlasmaNet.nnet.data.data_loaders as module_data
import PlasmaNet.nnet.model.loss as module_loss
import PlasmaNet.nnet.model.metric as module_metric
from PlasmaNet.nnet.parse_config import ConfigParser
from PlasmaNet.nnet.trainer.trainer import plot_batch
import PlasmaNet.nnet.model as module_arch

class PoissonNetwork(BasePoisson):
    """ Class for network solver of Poisson problem

    :param BasePoisson: Base class for Poisson routines
    """
    def __init__(self, cfg):
        """ Initialization of PoissonNetwork class

        :param cfg: config dictionnary
        :type cfg: dict
        """

        self.cfg_dl = ConfigParser(cfg)
        super().__init__(0, self.cfg_dl.lx, self.cfg_dl.nnx, 
                            0, self.cfg_dl.ly, self.cfg_dl.nny)
        self.res_train = self.cfg_dl.nnx

        # Load the network
        logger = self.cfg_dl.get_logger('test')

        # Setup data_loader instances
        data_loader = self.cfg_dl.init_obj('data_loader', module_data)
        self.alpha = data_loader.alpha
        self.scaling_factor = data_loader.scaling_factor

        # Build model architecture
        self.model = self.cfg_dl.init_obj('arch', module_arch)

        logger.info('Loading checkpoint: {} ...'.format(self.cfg_dl['resume']))
        checkpoint = torch.load(self.cfg_dl['resume'])
        state_dict = checkpoint['state_dict']
        if self.cfg_dl['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)

        # Prepare self.model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()    

    def solve(self, physical_rhs, res, Lx, Ly):
        """ Solve the Poisson problem with physical_rhs from neural network
        :param physical_rhs: - rho / epsilon_0 
        :type physical_rhs: ndtensor
        """
        ratio = self.alpha / (np.pi**2 / 4)**2 / (1 / Lx**2 + 1 / Ly**2)
        
        # Convert to torch.Tensor of shape (batch_size, 1, H, W) with normalization
        physical_rhs_torch = torch.from_numpy(self.physical_rhs[:, np.newaxis, :, :] 
                                                * ratio * self.scaling_factor).float().cuda()

        potential_torch = self.model(physical_rhs_torch)
        self.potentials = (self.res_train**2 / res**2 * potential_torch.detach().cpu().numpy()[:, 0] 
                                                / self.scaling_factor)
        