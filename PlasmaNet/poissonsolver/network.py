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

import PlasmaNet.nnet.data.data_loaders as module_data
import PlasmaNet.nnet.model as module_arch
from .base import BasePoisson
from ..nnet.parse_config import ConfigParser
from ..nnet.data.data_loaders import ratio_potrhs
from ..common.utils import create_dir

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
        # First initialize Base class
        if 'eval' in cfg:
            super().__init__(cfg['eval'])
        else:
            super().__init__(cfg['globals'])

        # Network configuration
        self.cfg_dl = ConfigParser(cfg)
        self.nnx_nn = self.cfg_dl.nnx

        # Logger
        logger = self.cfg_dl.get_logger('test')

        # Setup data_loader instances
        self.alpha = self.cfg_dl.alpha
        self.scaling_factor = self.cfg_dl.scaling_factor

        # Case specific properties
        self.ratio = ratio_potrhs(self.alpha, self.Lx, self.Ly)
        self.res_scale = self.nnx_nn**2 / self.nnx**2

        # Build model architecture
        self.model = self.cfg_dl.init_obj('arch', module_arch)

        # Load from directory, resume dir does not need to contain the full path to model_best.pth
        dir_resume = cfg['resume']
        dir_list = os.listdir(dir_resume)
        logger.info('Loading checkpoint: {} ...'.format(os.path.join(dir_resume, "model_best.pth")))
        checkpoint = torch.load(os.path.join(dir_resume, "model_best.pth"))
        #logger.info('Loading checkpoint: {} ...'.format(os.path.join(dir_resume, dir_list[-1], "model_best.pth")))
        #checkpoint = torch.load(os.path.join(dir_resume, dir_list[-1], "model_best.pth"))
        state_dict = checkpoint['state_dict']
        if self.cfg_dl['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)

        # Prepare self.model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
    
    def case_config(self, cfg: dict):
        """ Set the case configuration according to dict
        Reinitialize the base class

        :param cfg: configuration dict
        :type cfg: dict
        """
        super().__init__(cfg)
        self.ratio = ratio_potrhs(self.alpha, self.Lx, self.Ly)
        self.res_scale = self.nnx_nn**2 / self.nnx**2

    def solve(self, physical_rhs):
        """ Solve the Poisson problem with physical_rhs from neural network
        :param physical_rhs: - rho / epsilon_0 
        :type physical_rhs: ndtensor
        """
        self.physical_rhs = physical_rhs
        
        # Convert to torch.Tensor of shape (batch_size, 1, H, W) with normalization
        physical_rhs_torch = torch.from_numpy(self.physical_rhs[np.newaxis, np.newaxis, :, :] 
                                              * self.ratio * self.scaling_factor).float().cuda()

        # Apply the model
        potential_torch = self.model(physical_rhs_torch)

        # Retrieve the potential
        self.potential = self.res_scale / self.scaling_factor * potential_torch.detach().cpu().numpy()[0, 0]

    def run_case(self, case_dir: str, physical_rhs: np.ndarray, plot: bool, it=0):
        """ Run a Poisson linear system case

        :param case_dir: Case directory
        :type case_dir: str
        :param physical_rhs: physical rhs
        :type physical_rhs: np.ndarray
        :param Lx: length of the domain
        :type Lx: float
        :param plot: logical for plotting
        :type plot: bool
        """
        create_dir(case_dir)
        self.solve(physical_rhs)
        self.save(case_dir)
        if plot:
            fig_dir = case_dir + 'figures/'
            create_dir(fig_dir)
            self.plot_2D(fig_dir + '2D_{}'.format(it))
            self.plot_1D2D(fig_dir + 'full_{}'.format(it))