import re
import numpy as np
import scipy.constants as co

import torch
from cfdsolver import StreamerMorrow, PlasmaEuler


class StreamerMorrowDL(StreamerMorrow):
    """ Solve poisson with PlasmaNet. """
    def __init__(self, config):
        super().__init__(config)
        self.alpha = 0.1
        self.ratio = self.alpha / (np.pi**2 / 4)**2 / (1 / self.Lx**2 + 1 / self.Ly**2)

    def solve_poisson_dl(self, model):
        """ Solve poisson equation with PlasmaNet. """
        self.physical_rhs = (self.nd[1] - self.nd[0] - self.nd[2]) * co.e / co.epsilon_0
        # Convert to torch.Tensor of shape (batch_size, 1, H, W) with normalization
        physical_rhs_torch = torch.from_numpy(self.physical_rhs[np.newaxis, np.newaxis, :, :] 
                                                * self.ratio).float().cuda()
        potential_torch = model(physical_rhs_torch)
        potential_rhs = potential_torch.detach().cpu().numpy()[0, 0]
        self.potential = potential_rhs - self.backE * self.X


class PlasmaEulerDL(PlasmaEuler):
    """ Solve poisson with PlasmaNet. """
    def __init__(self, config, config_dl):
        super().__init__(config)
        self.alpha = 0.1
        self.ratio = self.alpha / (np.pi**2 / 4)**2 / (1 / self.Lx**2 + 1 / self.Ly**2)
        self.scaling_factor = 1.0e+6
        self.res_sim = config['mesh']['nnx']
        self.res_train = config_dl['globals']['nnx']

        if hasattr(self, 'globals'):
            self.globals['nnx_nn'] = self.res_train
            self.globals['Lx_nn'] = config_dl['globals']['lx']
            self.globals['arch'] = config_dl['arch']['type']
            
            re_casename = re.compile(r'.*/(\w*)/(\w*)/(\w*)')
            if re_casename.search(config_dl['resume']):
                self.globals['train_dataset'] = re_casename.search(config_dl['resume']).group(1)
            

    def solve_poisson_dl(self, model):
        """ Solve poisson equation with PlasmaNet. """
        self.physical_rhs = - (self.U[0] / self.m_e - self.n_back) * co.e / co.epsilon_0
        # Convert to torch.Tensor of shape (batch_size, 1, H, W) with normalization
        physical_rhs_torch = torch.from_numpy(self.physical_rhs[np.newaxis, np.newaxis, :, :] 
                                                * self.ratio * self.scaling_factor).float().cuda()
        potential_torch = model(physical_rhs_torch)
        potential_rhs = (self.res_train**2 / self.res_sim**2 * potential_torch.detach().cpu().numpy()[0, 0] 
                                                / self.scaling_factor)
        
        self.poisson.potential = potential_rhs
        self.E_field = self.poisson.E_field
    
        self.E_norm = np.sqrt(self.E_field[0]**2 + self.E_field[1]**2)
        if self.it == 1: self.E_max = np.max(self.E_norm)