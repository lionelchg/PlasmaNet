########################################################################################################################
#                                                                                                                      #
#                                           DL PoissonSolver with PlasmaNet                                            #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
from time import perf_counter

# From PlasmaNet
import PlasmaNet.nnet.data.data_loaders as module_data
import PlasmaNet.nnet.model as module_arch
import PlasmaNet.nnet.model.metric as module_metric
from ..nnet.parse_config import ConfigParser
from ..nnet.data.data_loaders import ratio_potrhs
from ..nnet.utils import MetricTracker
from ..nnet.trainer.trainer import plot_batch, plot_batch_Efield
from .base import BasePoisson


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
        self.alpha = self.cfg_dl.alpha
        self.scaling_factor = self.cfg_dl.scaling_factor

        # Case specific properties
        self.ratio = ratio_potrhs(self.alpha, self.Lx, self.Ly)
        self.res_scale = self.nnx_nn**2 / self.nnx**2

        # Build model architecture
        self.model = self.cfg_dl.init_obj('arch', module_arch)

        # Load from directory, resume dir does not need to contain the full path to model_best.pth
        path_resume = cfg['resume']
        self.logger.info(f'Loading checkpoint: {path_resume} ...')
        checkpoint = torch.load(path_resume)

        state_dict = checkpoint['state_dict']
        if self.cfg_dl['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)

        # Prepare self.model for testing
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.cfg_dl['n_gpu'] > 0 else 'cpu')
        self.model = self.model.to(self.device)
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
        if self.benchmark:
            #start_event = torch.cuda.Event(enable_timing=True)
            #end_event = torch.cuda.Event(enable_timing=True)
            #start_event.record()
            #torch.cuda.synchronize()
            comm_timer = perf_counter()
            total_timer = perf_counter()
        physical_rhs_torch = torch.from_numpy(self.physical_rhs[np.newaxis, np.newaxis, :, :]
                                              * self.ratio * self.scaling_factor).float().to(self.device)
        if self.benchmark:

            #end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded! 
            #comm_timer = start_event.elapsed_time(end_event)
            comm_timer = perf_counter() - comm_timer

        # Apply the model
        if self.benchmark:
            model_timer = perf_counter()
            #start_model = torch.cuda.Event(enable_timing=True)
            #end_model = torch.cuda.Event(enable_timing=True)
            #start_model.record()
            #torch.cuda.synchronize()

        potential_torch = self.model(physical_rhs_torch)
        if self.benchmark:
            #end_model.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded! 
            #model_timer = start_model.elapsed_time(end_model) 
            model_timer = perf_counter() - model_timer

        # Retrieve the potential
        if self.benchmark:
            #start_comm = torch.cuda.Event(enable_timing=True)
            #end_comm = torch.cuda.Event(enable_timing=True)
            #start_comm.record()
            #torch.cuda.synchronize()
            tmp = perf_counter()

        self.potential = self.res_scale / self.scaling_factor * potential_torch.detach().cpu().numpy()[0, 0]
        if self.benchmark:
            #end_comm.record()
            #torch.cuda.synchronize()  # Wait for the events to be recorded!
            #comm_timer_2 = start_comm.elapsed_time(end_comm)
            tmp = perf_counter() - tmp
            #comm_timer += comm_timer_2
            comm_timer += tmp
            #total_timer = start_event.elapsed_time(end_comm)
            total_timer = perf_counter() - total_timer

        # Print benchmarks
        if self.benchmark:
            self.logger.info(f"comm_timer={comm_timer}")
            self.logger.info(f"model_timer={model_timer}")
            self.logger.info(f"total_timer={total_timer}")

    def run_case(self, case_dir: Path, physical_rhs: np.ndarray, plot: bool, save=True):
        """ Run a Poisson linear system case

        :param case_dir: Case directory
        :type case_dir: Path
        :param physical_rhs: physical rhs
        :type physical_rhs: np.ndarray
        :param Lx: length of the domain
        :type Lx: float
        :param plot: logical for plotting
        :type plot: bool
        """
        case_dir.mkdir(parents=True, exist_ok=True)
        self.solve(physical_rhs)
        if save:
            self.save(case_dir)
        if plot:
            fig_dir = case_dir / 'figures'
            fig_dir.mkdir(parents=True, exist_ok=True)
            self.plot_2D(fig_dir / '2D', axis='off')
            self.plot_1D2D(fig_dir / 'full')

    def evaluate(self, data_dir, case_dir, plot):
        """
        Evaluate the network and returns the metrics of the network on the specified dataset
        """
        # Create case directory
        self.case_dir = Path(case_dir)
        if plot:
            self.fig_dir = self.case_dir / 'figures'
            self.fig_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        self.data_loader_cfg['args']['data_dir'] = data_dir
        data_loader = getattr(module_data, self.data_loader_cfg['type'])(self.cfg_dl, **self.data_loader_cfg['args'])
        self.metrics.reset()

        # Evaluate the network and follow metrics
        with torch.no_grad():
            for i, (data, target, data_norm, target_norm) in enumerate(tqdm(data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                data_norm, target_norm = data_norm.to(self.device), target_norm.to(self.device)

                # Divide input and outputs by scaling factor to go back to the real values
                output = self.res_scale / self.scaling_factor * self.model(data)
                target /= self.scaling_factor

                # Update MetricTracker with metrics
                for metric in self.metric_ftns:
                    self.metrics.update(metric.__name__, metric(output, target, self.cfg_dl).item())

                # Plot images if specified
                if plot:
                    #
                    # save sample images for the first images of the batch
                    #
                    fig = plot_batch(output, target, data, 0, i, self.cfg_dl)
                    fig.savefig(self.fig_dir / 'batch_{:05d}.png'.format(i), dpi=150, bbox_inches='tight')
                    fig = plot_batch_Efield(output, target, data, 0, i, self.cfg_dl)
                    fig.savefig(self.fig_dir / 'batch_Efield_{:05d}.png'.format(i), dpi=150, bbox_inches='tight')
                    plt.close()

        return self.metrics


class PoissonNetworkOpti(BasePoisson):
    """ Class for network solver of Poisson problem

    :param BasePoisson: Base class for Poisson routines

    """

    def __init__(self, cfg, model):
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
        # self.nnx_nn = self.cfg_dl.nnx
        self.nnx_nn = cfg['train_nnx']

        # Logger
        self.logger = self.cfg_dl.get_logger('poisson_nn')

        # Data loader if specified for batch evaluation
        if 'args' in cfg['data_loader']:
            self.data_loader_cfg = cfg['data_loader']
            self.metric_ftns = [getattr(module_metric, metric) for metric in cfg['metrics']]
            self.metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns])

        # Setup data_loader instances
        self.alpha = self.cfg_dl.alpha
        self.scaling_factor = self.cfg_dl.scaling_factor

        # Case specific properties
        self.ratio = ratio_potrhs(self.alpha, self.Lx, self.Ly)
        self.res_scale = self.nnx_nn**2 / self.nnx**2

        # Build model architecture
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.eval()

        # Load Evaluation data
        self.data_dir = cfg['data_loader']['args']['data_dir']

    def evaluateopti(self):
        """
        Evaluate the network and returns the metrics of the network on the specified dataset
        """

        # Load dataset
        self.data_loader_cfg['args']['data_dir'] = self.data_dir
        data_loader = getattr(module_data, self.data_loader_cfg['type'])(self.cfg_dl, **self.data_loader_cfg['args'])
        self.metrics.reset()

        # Evaluate the network and follow metrics
        with torch.no_grad():
            for i, (data, target, data_norm, target_norm) in enumerate(tqdm(data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                data_norm, target_norm = data_norm.to(self.device), target_norm.to(self.device)

                # Divide input and outputs by scaling factor to go back to the real values
                output = self.res_scale / self.scaling_factor * self.model(data)
                target /= self.scaling_factor

                # Update MetricTracker with metrics
                for metric in self.metric_ftns:
                    self.metrics.update(metric.__name__, metric(output, target, self.cfg_dl).item())

        return self.metrics._data.values[2, 2]
