########################################################################################################################
#                                                                                                                      #
#                                       Main class for Photo linear system solver                                      #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 25.09.2021                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import scipy.sparse.linalg as linalg
from time import perf_counter
from scipy.sparse.linalg import spsolve
import copy
import os
from pathlib import Path
import yaml
import torch

from .photo import photo_axisym, lambda_j_two, lambda_j_three, A_j_two, A_j_three
from .base import BasePhoto
from ..poissonsolver.linsystem import impose_dirichlet
from ..nnet.parse_config import ConfigParser

import PlasmaNet.nnet.model as module_arch
import PlasmaNet.nnet.model.metric as module_metric
from ..nnet.utils import MetricTracker

def prepare_model(model, device, path_resume, logger, cfg_dl):
    """ Load weights from path_resume and prepare model for evaluation """
        # Load from directory, resume dir does not need to contain the full path to model_best.pth
    logger.info(f'Loading checkpoint: {path_resume} ...')
    checkpoint = torch.load(path_resume)

    state_dict = checkpoint['state_dict']
    if cfg_dl['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Prepare model for testing
    model = model.to(device)
    model.eval()

class PhotoNetwork(BasePhoto):
    """ Class for neural network solver of Photoionization source term

    :param BasePhoto: Base class for Poisson routines
    """
    def __init__(self, cfg):
        super().__init__(cfg['globals'])

        self.scale = self.dx * self.dy
        self.photo_model = cfg['photo_model']

        # Pressure in Torr
        self.pO2 = 150

        # Two or three helmholtz equations depending on the photoionization model
        self.jtot = 2
        self.Sphj1 = np.zeros_like(self.Sph)
        self.Sphj2 = np.zeros_like(self.Sph)

        if self.photo_model == 'three':
            self.jtot = 3
            self.Sphj3 = np.zeros_like(self.Sph)

        # Load network
        # Architecture parsing in database
        if 'db_file' in cfg['arch']:
            with open(Path(os.getenv('ARCHS_DIR')) / cfg['arch']['db_file']) as yaml_stream:
                archs = yaml.safe_load(yaml_stream)
            tmp_cfg_arch = archs[cfg['arch']['name']]
            if 'args' in cfg['arch']:
                tmp_cfg_arch['args'] = {**cfg['arch']['args'], **tmp_cfg_arch['args']}
            cfg['arch'] = tmp_cfg_arch

        # Network configuration
        self.cfg_dl = ConfigParser(cfg)
        self.nnx_nn = self.cfg_dl.nnx

        # Logger
        self.logger = self.cfg_dl.get_logger('poisson_nn')

        # Data loader if specified for batch evaluation
        if 'args' in cfg['data_loader']:
            self.data_loader_cfg = cfg['data_loader']
            self.metric_ftns = [getattr(module_metric, metric) for metric in cfg['metrics']]
            self.metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns])

        # Setup data_loader instances
        self.scaling_factor = self.cfg_dl.scaling_factor

        # Normalization for jtot = 2 (hardcoded)
        self.norm_j1 = 1.0e+2

        # Case specific properties
        self.res_scale = self.nnx_nn**2 / self.nnx**2

        # Build model architecture
        self.model_j1 = self.cfg_dl.init_obj('arch', module_arch)
        self.model_j2 = self.cfg_dl.init_obj('arch', module_arch)
        if self.jtot == 3: self.model_j3 = self.cfg_dl.init_obj('arch', module_arch)
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.cfg_dl['n_gpu'] > 0 else 'cpu')

        prepare_model(self.model_j1, self.device, cfg['resume'][0], self.logger, self.cfg_dl)
        prepare_model(self.model_j2, self.device, cfg['resume'][1], self.logger, self.cfg_dl)
        if self.jtot == 3: prepare_model(self.model_j3, self.device, cfg['resume'][2], self.logger, self.cfg_dl)

    def solve(self, ioniz_rate: np.ndarray):
        """ Solve the Poisson equation with ioniz_rate as charge density / epsilon_0

        :param ioniz_rate: - rho / epsilon_0
        :type ioniz_rate: np.ndarray
        :param bcs: Dictionnary of boundary conditions
        :type bcs: dict
        """
        self.ioniz_rate = ioniz_rate
        if self.photo_model == 'two':
            for i in range(2):
                rhs = self.ioniz_rate
                rhs_torch = torch.from_numpy(rhs[np.newaxis, np.newaxis, :, :]
                        * self.scaling_factor).float().to(self.device)
                if i == 0:
                    Sphj1_torch = self.model_j1(rhs_torch)
                    self.Sphj1 = Sphj1_torch.detach().cpu().numpy()[0, 0, :, :] / self.scaling_factor / self.norm_j1
                elif i == 1:
                    Sphj2_torch = self.model_j2(rhs_torch)
                    self.Sphj2 = Sphj2_torch.detach().cpu().numpy()[0, 0, :, :] / self.scaling_factor
            self.Sph = self.Sphj1 + self.Sphj2
            self.Sph = np.maximum(1e20, self.Sph)

