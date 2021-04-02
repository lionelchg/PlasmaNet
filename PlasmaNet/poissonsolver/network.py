########################################################################################################################
#                                                                                                                      #
#                                           DL PoissonSolver with PlasmaNet                                            #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os
import numpy as np
import torch

from .base import BasePoisson
import PlasmaNet.nnet.data.data_loaders as module_data
from PlasmaNet.nnet.parse_config import ConfigParser
import PlasmaNet.nnet.model as module_arch


class PoissonNetwork(BasePoisson):
    """ Class for network solver of Poisson problem

    :param BasePoisson: Base class for Poisson routines
    """
    def __init__(self, cfg):
        """
        Initialization of PoissonNetwork class

        :param cfg: config dictionary
        :type cfg: dict
        """
        # First copy values for initialization of base class
        cfg['xmin'], cfg['xmax'], cfg['nnx'] = 0, cfg['globals']['lx'], cfg['globals']['nnx']
        cfg['ymin'], cfg['ymax'], cfg['nny'] = 0, cfg['globals']['ly'], cfg['globals']['nny']
        super().__init__(cfg)

        self.cfg_dl = ConfigParser(cfg)
        self.res_train = self.cfg_dl.nnx

        # Load the network
        logger = self.cfg_dl.get_logger('test')

        # Setup data_loader instances
        self.alpha = 0.1
        self.scaling_factor = self.cfg_dl.scaling_factor

        # Build model architecture
        self.model = self.cfg_dl.init_obj('arch', module_arch)

        # Load from directory, resume dir does not need to contain the full path to model_best.pth
        dir_resume = cfg['resume']
        dir_list = os.listdir(dir_resume)
        logger.info('Loading checkpoint: {} ...'.format(os.path.join(dir_resume, dir_list[-1], "model_best.pth")))
        checkpoint = torch.load(os.path.join(dir_resume, dir_list[-1], "model_best.pth"))
        state_dict = checkpoint['state_dict']
        if self.cfg_dl['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)

        # Prepare self.model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()    

    def solve(self, physical_rhs, Lx):
        """ Solve the Poisson problem with physical_rhs from neural network
        :param physical_rhs: - rho / epsilon_0 
        :type physical_rhs: ndtensor
        """
        self.physical_rhs = physical_rhs
        res = self.physical_rhs.shape[-1]
        ratio = self.alpha / (np.pi**2 / 4)**2 / (2 / Lx**2)
        
        # Convert to torch.Tensor of shape (batch_size, 1, H, W) with normalization
        physical_rhs_torch = torch.from_numpy(self.physical_rhs[np.newaxis, np.newaxis, :, :] 
                                              * ratio * self.scaling_factor).float().cuda()

        potential_torch = self.model(physical_rhs_torch)
        self.potential = (self.res_train**2 / res**2 * potential_torch.detach().cpu().numpy()[0, 0] 
                           / self.scaling_factor)
