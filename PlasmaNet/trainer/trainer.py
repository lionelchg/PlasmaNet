########################################################################################################################
#                                                                                                                      #
#                                                    Trainer class                                                     #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 10.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import torch

from ..base import BaseTrainer
from .plot import plot_batch, plot_distrib, plot_scales
from ..utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class.
    Train a model for the given criterion and optimizer. Metrics are computed at each epoch. A scheduler may be used,
    as well as a different validation dataloader than the training dataloader.
    If len_epoch is specified, iteration-based training is used, otherwise epoch-based.
    """
    def __init__(self, model, criterion, metric_fnts, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_fnts, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *self.criterion.loss_list,
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *self.criterion.loss_list,
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training method for the specified epoch.
        Returns a log that contains the average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, data_norm, target_norm) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            data_norm, target_norm = data_norm.to(self.device), target_norm.to(self.device)

            self.optimizer.zero_grad()

            output_raw = self.model(data, epoch)
            if output_raw.size(1) != 1:
                output = output_raw[:,0].unsqueeze(1)
                multiple_outputs = True
            else:
                output = output_raw

            if self.criterion.require_input_data():
                loss = self.criterion(output, target, data=data, target_norm=target_norm, data_norm=data_norm)
            else:
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Update MetricTracker
            for key, value in self.criterion.log().items():
                self.train_metrics.update(key, value.item())
            for metric in self.metric_ftns:
                self.train_metrics.update(metric.__name__, metric(output, target, self.config).item())

            # Writer logger output
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:06e}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            # Set writer step with epoch
            if batch_idx == 0:
                self.writer.set_step(epoch - 1)
            # Figure output after the 1st batch of each epoch
            if epoch % self.config['trainer']['plot_period'] == 0 and batch_idx == 0:
                self._batch_plots(output, target, data, epoch, batch_idx)
                if multiple_outputs:
                    # Plot input, target, output and residual
                    fig = plot_scales(output_raw, target, data, epoch, batch_idx, self.config)
                    fig.savefig(self.config.fig_dir / 'train_scales_{:05d}.png'.format(epoch), dpi=150, bbox_inches='tight')
                    self.writer.add_figure('Different Scales', fig)



            if batch_idx == self.len_epoch:  # Break iteration-based training
                break

        # Extract averages and send TensorBoard
        log = self.train_metrics.result()
        # Add current learning rate to log
        for group in self.optimizer.param_groups:
            log['LearningRate'] = group['lr']
        self._send_log_to_tb(log)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if 'ReduceLROnPlateau' in str(self.lr_scheduler.__class__):
                # ReduceLROnPlateau needs a metric for each step
                self.lr_scheduler.step(log[self.config['lr_scheduler']['plateau_metric']])
            else:
                self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """ Validate after training. Returns a log with validation information. """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target, data_norm, target_norm) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                data_norm, target_norm = data_norm.to(self.device), target_norm.to(self.device)

                output_raw = self.model(data, epoch)
                if output_raw.size(1) != 1:
                    output = output_raw[:,0].unsqueeze(1)
                    multiple_outputs = True
                else:
                    output = output_raw

                if self.criterion.require_input_data():
                    loss = self.criterion(output, target, data=data, target_norm=target_norm, data_norm=data_norm)
                else:
                    loss = self.criterion(output, target)

                # Update MetricTracker
                for key, value in self.criterion.log().items():
                    self.valid_metrics.update(key, value.item())
                for metric in self.metric_ftns:
                    self.valid_metrics.update(metric.__name__, metric(output, target, self.config).item())
                # Set writer step with epoch
                if batch_idx == 0:
                    self.writer.set_step(epoch - 1, 'valid')
                # Figure output after the 1st batch of each epoch
                if epoch % self.config['trainer']['plot_period'] == 0 and batch_idx == 0:
                    self._batch_plots(output, target, data, epoch, batch_idx, 'valid')
                    if multiple_outputs:
                        # Plot input, target, output and residual
                        fig = plot_scales(output_raw, target, data, epoch, batch_idx, self.config)
                        fig.savefig(self.config.fig_dir / 'val_scales_{:05d}.png'.format(epoch), dpi=150, bbox_inches='tight')
                        self.writer.add_figure('Different Scales', fig)


        # Add histogram of model parameters to the TensorBoard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        # Extract averages and send to TensorBoard
        val_log = self.valid_metrics.result()
        self._send_log_to_tb(val_log)

        return val_log

    def _batch_plots(self, output, target, data, epoch, batch_idx, mode='train'):
        """ Plots to realise during training and validation loops. File and TensorBoard output. """
        # Plot input, target, output and residual
        fig = plot_batch(output, target, data, epoch, batch_idx, self.config)
        fig.savefig(self.config.fig_dir / '{}_{:05d}.png'.format(mode, epoch), dpi=150, bbox_inches='tight')
        self.writer.add_figure('ComparisonWithResiduals', fig)
        # Plot output vs target distribution
        fig = plot_distrib(output, target, epoch, batch_idx)
        fig.savefig(self.config.fig_dir / '{}_distrib_{:05d}.png'.format(mode, epoch), dpi=150, bbox_inches='tight')
        self.writer.add_figure('OutputTargetDistribution', fig)

    def _send_log_to_tb(self, log):
        """ Send log to TensorBoard. """
        # Adding train or valid to tag is managed by the TensorboardWriter class based on the last set_step() specified
        # mode ('train' by default, or 'valid')
        for key, value in log.items():
            if key in self.criterion.loss_list:
                self.writer.add_scalar('ComposedLosses/' + key, value)
            elif key in [metric.__name__ for metric in self.metric_ftns]:
                self.writer.add_scalar('Metrics/' + key, value)
            else:
                self.writer.add_scalar(key, value)

    def _progress(self, batch_idx):
        """ A simple progress meter. """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


