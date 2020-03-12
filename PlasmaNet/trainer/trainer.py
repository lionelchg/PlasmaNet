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
import matplotlib.pyplot as plt


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

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

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
                if self.data_loader.normalize:
                    loss = self.criterion(output, target, data, target_norm, data_norm)
                else:
                    loss = self.criterion(output, target, data)
            else:
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Update log
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for metric in self.metric_ftns:
                self.train_metrics.update(metric.__name__, metric(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:06f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            # Figure output after the 1st batch of each epoch
            if batch_idx == 0:
                # self.writer.add_image('input', make_grid(data[:10].cpu(), nrow=1, normalize=True))
                # self.writer.add_image('output', make_grid(output[:10].cpu(), nrow=1, normalize=True))
                # self.writer.add_image('target', make_grid(target[:10].cpu(), nrow=1, normalize=True))
                self.writer.add_figure('train', plot_numpy(output, target, data, epoch, batch_idx))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k : v for k, v in val_log.items()})

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
                    if self.data_loader.normalize:
                        loss = self.criterion(output, target, data, target_norm, data_norm)
                    else:
                        loss = self.criterion(output, target, data)
                else:
                    loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for metric in self.metric_ftns:
                    self.valid_metrics.update(metric.__name__, metric(output, target))
                # Figure output after the 1st batch of each epoch
                if batch_idx == 0:
                    # self.writer.add_image('val_input', make_grid(data[:10].cpu(), nrow=1, normalize=True))
                    # self.writer.add_image('val_output', make_grid(output[:10].cpu(), nrow=1, normalize=True))
                    # self.writer.add_image('val_target', make_grid(target[:10].cpu(), nrow=1, normalize=True))
                    self.writer.add_figure('valid',plot_numpy(output, target, data, epoch, batch_idx))

        # Add histogram of model parameters to the TensorBoard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()

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
    fig, axes = plt.subplots(figsize=(12, 25), nrows=5, ncols=3)
    fig.suptitle(' Model {} for epoch {}'.format(batch_idx, epoch))

    for k in range(5):
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
    return fig
