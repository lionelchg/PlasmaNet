########################################################################################################################
#                                                                                                                      #
#                                              Configuration file parser                                               #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 10.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import logging
import os
from datetime import datetime
from functools import reduce, partial
from operator import getitem
from pathlib import Path
from copy import deepcopy

import numpy as np

from .logger import setup_logging
from .utils import read_yaml, write_yaml


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=''):
        """
        Class to parse a yaml configuration file. Handles hyperparameters for training, initialisations of modules,
        checkpoint saving and logging module.

        Parameters
        ----------
        config : dict()
            Dictionary containing configurations and hyperparameters for training. E.g. contents of `config.yml` file.

        resume : str()
            Path to the checkpoint to load to resume training.

        modification : dict() keychain:value
            Specifying values to be replaced from config dictionary.

        run_id
            Unique identifier for training processes. Used to save checkpoints and training log. Default: timestamp
        """
        # Load config file and apply modifications
        self._config = _update_config(config, modification)
        self.resume = resume

        if 'trainer' in self.config:
            # Set save_dir where trained model and log will be saved
            save_dir = Path(self.config['trainer']['save_dir'])

            exper_name = self.config['name']
            if run_id is None:  # use timestamp as default run-id
                run_id = datetime.now().strftime(r'%m%d_%H%M%S')
            self._save_dir = save_dir / 'models' / exper_name / run_id
            self._log_dir = save_dir / 'log' / exper_name / run_id
            self._fig_dir = save_dir / 'figures' / exper_name / run_id

            # Make directories for checkpoints saving and logs
            exist_ok = run_id == ''  # if True, mkdir ignores FilesExistsError (similar to `mkdir -p`)
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.fig_dir.mkdir(parents=True, exist_ok=exist_ok)

            # Save updated config file to the checkpoint directory
            write_yaml(self.config, self.save_dir / 'config.yml')
        else:
            # For evaluation (no trainer entry in .yml)
            if 'casename' in self.config:
                self._log_dir = Path(self.config['casename'])
                self.log_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = Path(self.config['name'])
                self._fig_dir = save_dir / 'figures'
                self.fig_dir.mkdir(parents=True, exist_ok=True)
                self._log_dir = save_dir
                self.log_dir.mkdir(parents=True, exist_ok=True)
                write_yaml(self.config, save_dir / 'config.yml')

        # Number of channels of the network
        # self.channels = self.config['arch']['args']['data_channels']
        self.channels = self.config['data_loader']['data_channels']

        # data_loader related parameters
        if 'data_loader' in self.config:
            if 'args' in self.config['data_loader']:
                self.batch_size = self.config['data_loader']['args']['batch_size']
                self.guess = self.config['data_loader']['args'].get('guess')
                self.modes = self.config['data_loader']['args'].get('modes')

            self.scaling_factor = self.config['data_loader']['args']['scaling_factor']
            if self.config['data_loader']['type'] == 'PoissonDataLoader':
                # Length and resolution invariance parameters
                self.normalization = self.config['data_loader']['args']['normalize']
                self.alpha = self.config['data_loader']['args']['alpha']

        # Declare global runtime parameters attributes
        self.nnx = self.config['globals']['nnx']
        self.nny = self.config['globals']['nny']
        self.xmin, self.xmax = self.config['globals']['xmin'], self.config['globals']['xmax']
        self.ymin, self.ymax = self.config['globals']['ymin'], self.config['globals']['ymax']

        # Compute quantities of interest
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
        self.dx = self.Lx / (self.nnx - 1)
        self.dy = self.Ly / (self.nny - 1)
        self.ds = self.dx * self.dy
        self.surface = self.Lx * self.Ly
        x = np.linspace(self.xmin, self.xmax, self.nnx)
        y = np.linspace(self.ymin, self.ymax, self.nny)
        self.X, self.Y = np.meshgrid(x, y)

        self.coord = self.config['globals']['coord']
        if self.coord == 'cyl':
            r_nodes = deepcopy(self.Y)
            r_nodes[0] = self.dy / 4
            self.r_nodes = r_nodes
        else:
            self.r_nodes = None

        # If adaptative weights and gradient plots
        if 'adaptative' in config['loss']['args']:
            self.adaptative = True
            self.adaptative_weight = config['loss']['args']['adaptative']

        if 'gradients' in config['loss']['args']:
            self.gradients = True
            self.gradients_every = config['loss']['args']['gradients']
        else:
            self.gradients = False

        # Configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """ Initializes this class from some CLI argparse arguments. Use in train, test, etc. """
        for option in options:
            args.add_argument(*option.flags, default=None, type=option.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.yml'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.yml', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_yaml(cfg_fname)
        if args.config and resume:
            # Update new config for fine-tuning
            config.update(read_yaml(args.config))

        # Parse custom CLI options into dictionary
        modification = {option.target: getattr(args, _get_opt_name(option.flags)) for option in options}

        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given in the [name]['type'] field in the config file, in the given module,
        and returns the instance initialized with the corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        if 'pipe_config' in self[name] and self[name]['pipe_config']:  # Object requires config at instantiation
            return getattr(module, module_name)(self, *args, **module_args)
        else:
            return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given in the [name]['type'] field in the config file, in the given module,
        and returns the function with given arguments fixed with functools.partial

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """ Access items like ordinary dictionary. """
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'Verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # Setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def fig_dir(self):
        return self._fig_dir


# Helper functions to update config dict with custom CLI options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flag in flags:
        if flag.startswith('--'):
            return flag.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """ Set a value in a nested object in tree by a sequence of keys. """
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """ Access a nested object in tree by a sequence of keys. """
    return reduce(getitem, keys, tree)
