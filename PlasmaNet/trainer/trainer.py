########################################################################################################################
#                                                                                                                      #
#                                                    Trainer class                                                     #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 10.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import torch
from torchvision.utils import make_grid
from ..base import BaseTrainer
from ..utils import inf_loop, MetricTracker
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


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
            output = self.model(data)
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
                self.train_metrics.update(metric.__name__, metric(output, target).item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:06f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            # Set writer step with epoch
            if batch_idx == 0:
                self.writer.set_step(epoch - 1)
            # Figure output after the 1st batch of each epoch
            if epoch % self.config['trainer']['plot_period'] == 0 and batch_idx == 0:
                self.writer.add_figure('ComparisonWithResiduals', plot_numpy(output, target, data, epoch, batch_idx))

            if batch_idx == self.len_epoch:  # Break iteration-based training
                break

        # Extract averages and send TensorBoard
        log = self.train_metrics.result()
        for key, value in log.items():
            self.writer.add_scalar(key, value)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
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

                output = self.model(data)
                if self.criterion.require_input_data():
                    loss = self.criterion(output, target, data=data, target_norm=target_norm, data_norm=data_norm)
                else:
                    loss = self.criterion(output, target)

                # Update MetricTracker
                for key, value in self.criterion.log().items():
                    self.train_metrics.update(key, value.item())
                for metric in self.metric_ftns:
                    self.valid_metrics.update(metric.__name__, metric(output, target).item())
                # Set writer step with epoch
                if batch_idx == 0:
                    self.writer.set_step(epoch - 1, 'valid')
                # Figure output after the 1st batch of each epoch
                if epoch % self.config['trainer']['plot_period'] == 0 and batch_idx == 0:
                    self.writer.add_figure('ComparisonWithResiduals', plot_numpy(output, target, data, epoch, batch_idx))

        # Add histogram of model parameters to the TensorBoard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        # Extract averages and send to TensorBoard
        val_log = self.valid_metrics.result()
        for key, value in val_log.items():
            self.writer.add_scalar(key, value)

        return val_log

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


def plot_numpy(output, target, data, epoch, batch_idx):
    """ Matplotlib plots. """
    # Detach tensors and send them to cpu as numpy
    data_np = data.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()

    # Lots of plots
    fig, axes = plt.subplots(figsize=(12, 25), nrows=5, ncols=4)
    fig.suptitle(' Epoch {} batch_idx {}'.format(epoch, batch_idx))

    for k in range(5):  # First 5 items of the batch
        tt = axes[k, 0].imshow(data_np[batch_idx + k, 0], origin='lower')
        axes[k, 0].set_title('rhs')
        axes[k, 0].axis('off')
        fig.colorbar(tt, ax=axes[k, 0])

        tt = axes[k, 1].imshow(output_np[batch_idx + k, 0], origin='lower')
        axes[k, 1].set_title('predicted potential')
        axes[k, 1].axis('off')
        fig.colorbar(tt, ax=axes[k, 1])

        tt = axes[k, 2].imshow(target_np[batch_idx + k, 0], origin='lower')
        axes[k, 2].set_title('target potential')
        axes[k, 2].axis('off')
        fig.colorbar(tt, ax=axes[k, 2])

        tt = axes[k, 3].imshow(np.abs(target_np[batch_idx + k, 0] - output_np[batch_idx + k, 0]), origin='lower')
        axes[k, 3].set_title('residual')
        axes[k, 3].axis('off')
        fig.colorbar(tt, ax=axes[k, 3])
    return fig
