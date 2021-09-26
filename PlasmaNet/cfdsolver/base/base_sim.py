########################################################################################################################
#                                                                                                                      #
#                                                    BaseSim class                                                     #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 05.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import re
import yaml
import logging
from pathlib import Path

from .metric import Grid
from ...common.utils import create_dir

class BaseSim(Grid):
    def __init__(self, config):
        super().__init__(config)
        self.number = 1
        self.save_type = config['output']['save']
        self.verbose = config['output']['verbose']
        self.period = config['output']['period']

        # Saving of solutions data and/or figures
        self.case_dir = Path(config['casename'])
        self.case_dir.mkdir(parents=True, exist_ok=True)

        self.file_type = config['output']['files']
        self.save_fig, self.save_data = re.search('fig', self.file_type), re.search('data', self.file_type)
        if self.save_data:
            self.data_dir = self.case_dir / 'data/'
            self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.save_fig:
            self.fig_dir = self.case_dir / 'figures/'
            self.fig_dir.mkdir(parents=True, exist_ok=True)

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
        if 'nit' in config['params']:
            self.nit = config['params']['nit']
        elif 'end_time' in config['params']:
            self.end_time = config['params']['end_time']

        # Logging file
        if 'loglevel' in config['output']:
            self.loglevel = eval(f"logging.{config['output']['loglevel'].upper()}")
        else:
            self.loglevel = logging.INFO
        logging.basicConfig(filename=self.case_dir / 'run.log',
            level=self.loglevel,
            format='%(asctime)s %(message)s',
            datefmt='%m-%d %H:%M')

        # Dump the configuration file in the case
        with open(self.case_dir / 'config.yml', 'w') as file:
            yaml.dump(config, file)

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
                    logging.info('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, self.dt, self.dtsum, width=14))
                if self.save_fig: self.plot()
                if self.save_data: self.save()
                self.number += 1
        elif self.save_type == 'time':
            if np.abs(self.dtsum - self.number * self.period) < 0.5 * self.dt or it == self.nit:
                if self.verbose:
                    logging.info('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, self.dt, self.dtsum, width=14))
                if self.save_fig: self.plot()
                if self.save_data: self.save()
                self.number += 1
        elif self.save_type == 'none':
            pass
