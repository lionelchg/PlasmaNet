########################################################################################################################
#                                                                                                                      #
#                                                    BaseSim class                                                     #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 05.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import re

from .metric import Grid
from ..utils import create_dir


class BaseSim(Grid):
    def __init__(self, config):
        super().__init__(config)
        self.number = 1
        self.save_type = config['output']['save']
        self.verbose = config['output']['verbose']
        self.period = config['output']['period']

        # Saving of solutions data and/or figures
        self.file_type = config['output']['files']
        self.save_fig, self.save_data = re.search('fig', self.file_type), re.search('data', self.file_type)
        if self.save_data:
            self.data_dir = 'data/' + config['casename']
            create_dir(self.data_dir)
        if self.save_fig:
            self.fig_dir = 'figures/' + config['casename']
            create_dir(self.fig_dir)

        self.dtsum = 0
        self.dt = 0
        self.it = 0

        # Timestep calculation through cfl
        if 'cfl' in config['params']:
            self.cfl = config['params']['cfl']

        # Imposed timestep
        if 'dt' in config['params']:
            self.dt = config['params']['dt']

        # Number of iterations
        self.nit = config['params']['nit']

    def plot(self):
        """ Abstract method for plotting """
        raise NotImplementedError

    def save(self):
        """ Abstract method for saving data """
        raise NotImplementedError

    def postproc(self, it):
        if self.save_type == 'iteration':
            if it % self.period == 0 or it == self.nit:
                if self.verbose:
                    print('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, self.dt, self.dtsum, width=14))
                if self.save_fig: self.plot()
                if self.save_data: self.save()
                self.number += 1
        elif self.save_type == 'time':
            if np.abs(self.dtsum - self.number * self.period) < 0.5 * self.dt or it == self.nit:
                if self.verbose:
                    print('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, self.dt, self.dtsum, width=14))
                if self.save_fig: self.plot()
                if self.save_data: self.save()
                self.number += 1
        elif self.save_type == 'none':
            pass
