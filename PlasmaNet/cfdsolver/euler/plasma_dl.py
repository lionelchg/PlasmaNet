########################################################################################################################
#                                                                                                                      #
#                               Plasma Euler equations solver with DL Poisson solver                                   #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 19.03.2021                                        #
#                                                                                                                      #
########################################################################################################################

import re
import os

import torch
import numpy as np
import scipy.constants as co
from time import perf_counter

from .plasma import PlasmaEuler
from PlasmaNet.nnet import ConfigParser
import PlasmaNet.nnet.model as module_arch


class PlasmaEulerDL(PlasmaEuler):
    """
    Solve Poisson with PlasmaNet.
    Inherits PlasmaEuler and overloads the solve_poisson method with the PlasmaNet call to a given model.
    """

    def __init__(self, config, config_dl, log_perf=False):
        super().__init__(config)
        self.alpha = 0.1
        self.ratio = self.alpha / (np.pi**2 / 4)**2 / (1 / self.Lx**2 + 1 / self.Ly**2)
        self.scaling_factor = 1.0e+6
        self.res_sim = config['mesh']['nnx']
        self.res_train = config_dl['globals']['nnx']

        if hasattr(self, 'globals'):
            self.globals['nnx_nn'] = self.res_train
            self.globals['Lx_nn'] = config_dl['globals']['Lx']
            self.globals['arch'] = config_dl['arch']['type']

            re_casename = re.compile(r'.*/(\w*)/(\w*)/(\w*)')
            if re_casename.search(config_dl['resume']):
                self.globals['train_dataset'] = re_casename.search(config_dl['resume']).group(1)

        # Initialise inference network
        cfg_dl = ConfigParser(config_dl)
        self.logger = cfg_dl.get_logger("test")
        self.logger.info(config_dl["casename"])
        self.model = cfg_dl.init_obj("arch", module_arch)

        # Load checkpoint
        dir_list = os.listdir(cfg_dl["resume"])
        checkpoint_path = os.path.join(cfg_dl["resume"], dir_list[-1], "model_best.pth")
        self.logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        if cfg_dl["n_gpu"] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(checkpoint["state_dict"])

        # Prepare model for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()

        # Log perf for benchmark
        self.log_perf = log_perf

    def solve_poisson(self):
        """ Solve poisson equation with neural network model (pytorch object) """
        poisson_timer = perf_counter()

        self.physical_rhs = - (self.U[0] / self.m_e - self.n_back) * co.e / co.epsilon_0
        # Convert to torch.Tensor of shape (batch_size, 1, H, W) with normalization
        comm_timer = perf_counter()

        physical_rhs_torch = torch.from_numpy(self.physical_rhs[np.newaxis, np.newaxis, :, :]
                                              * self.ratio * self.scaling_factor).float().cuda()

        comm_timer = perf_counter() - comm_timer
        network_timer = perf_counter()

        potential_torch = self.model(physical_rhs_torch)

        network_timer = perf_counter() - network_timer
        comm_timer += perf_counter()

        potential_rhs = (self.res_train**2 / self.res_sim**2 * potential_torch.detach().cpu().numpy()[0, 0]
                         / self.scaling_factor)

        comm_timer = perf_counter() - comm_timer

        self.poisson.potential = potential_rhs
        self.E_field = self.poisson.E_field

        self.E_norm = np.sqrt(self.E_field[0]**2 + self.E_field[1]**2)
        if self.it == 1:
            self.E_max = np.max(self.E_norm)

        poisson_timer = perf_counter() - poisson_timer
        if self.log_perf:
            self.logger.info("Poisson DL perf: {}".format(poisson_timer))
            self.logger.info("Poisson comm perf: {}".format(comm_timer))
            self.logger.info("Poisson network perf: {}".format(network_timer))
