########################################################################################################################
#                                                                                                                      #
#                                                  Plots for trainer                                                   #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 02.04.2020                                        #
#                                                                                                                      #
########################################################################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ..operators.gradient import grad
from ...common.plot import plot_ax_scalar, plot_ax_vector_arrow, round_up

matplotlib.use('Agg')


def plot_batch(output, target, data, epoch, batch_idx, config):
    """ Matplotlib plots of potential/RHS during training """
    # Detach tensors and send them to cpu as numpy
    data_np = data.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()


    # Lots of plots
    fig, axes = plt.subplots(figsize=(16, 12), nrows=4, ncols=4, sharex=True, sharey=True)
    fig.suptitle('Epoch {} batch {}'.format(epoch, batch_idx), fontsize=16, y=0.95)

    for k in range(4):  # First 4 items of the batch
        if config.channels == 3:
            data_tmp = data_np[batch_idx + k, 2]
        elif config.channels == 2:
            data_tmp = data_np[batch_idx + k, 1]
        else:
            data_tmp = data_np[batch_idx + k, 0]
        output_tmp = output_np[batch_idx + k, 0]
        target_tmp = target_np[batch_idx + k, 0]
        # Same scale for output and target
        target_max = round_up(np.max(np.abs(target_tmp)), decimals=1)

        # Apply title only on first line
        if k == 0:
            plot_ax_scalar(fig, axes[k, 0], config.X, config.Y, data_tmp, 'RHS')
            plot_ax_scalar(fig, axes[k, 1], config.X, config.Y, output_tmp, 'Predicted potential', max_value=target_max)
            plot_ax_scalar(fig, axes[k, 2], config.X, config.Y, target_tmp, 'Target potential', max_value=target_max)
            plot_ax_scalar(fig, axes[k, 3], config.X, config.Y, np.abs(target_tmp - output_tmp), 'Residual',
                        cmap='Blues')
        else:
            plot_ax_scalar(fig, axes[k, 0], config.X, config.Y, data_tmp, '')
            plot_ax_scalar(fig, axes[k, 1], config.X, config.Y, output_tmp, '', max_value=target_max)
            plot_ax_scalar(fig, axes[k, 2], config.X, config.Y, target_tmp, '', max_value=target_max)
            plot_ax_scalar(fig, axes[k, 3], config.X, config.Y, np.abs(target_tmp - output_tmp), '',
                        cmap='Blues')

        if data.size(1)==2:
            lamb_val = data[0, 1, 0, 0]
            fig.suptitle(f'Lambda value {lamb_val}')

    return fig

def plot_batch_Efield(output, target, data, epoch, batch_idx, config):
    """ Matplotlib plots of electric field during training """
    # Detach tensors and send them to cpu as numpy
    data_np = data.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()

    # Lots of plots
    fig, axes = plt.subplots(figsize=(16, 12), nrows=4, ncols=4, sharex=True, sharey=True)
    fig.suptitle('Epoch {} batch {}'.format(epoch, batch_idx), fontsize=16, y=0.95)

    for k in range(4):  # First 4 items of the batch
        if config.channels == 3:
            data_tmp = data_np[batch_idx + k, 2]
        elif config.channels == 2:
            data_tmp = data_np[batch_idx + k, 1]
        else:
            data_tmp = data_np[batch_idx + k, 0]
        output_tmp = - grad(output_np[batch_idx + k, 0], config.dx, config.dy)
        target_tmp = - grad(target_np[batch_idx + k, 0], config.dx, config.dy)
        norm_output_tmp = np.sqrt(output_tmp[0]**2 + output_tmp[1]**2)
        norm_target_tmp = np.sqrt(target_tmp[0]**2 + target_tmp[1]**2)
        target_max = round_up(np.max(np.abs(norm_target_tmp)), decimals=1)
        # Apply title only on first line
        if k == 0:
            plot_ax_scalar(fig, axes[k, 0], config.X, config.Y, data_tmp, 'RHS')
            # Same scale for output and target
            plot_ax_vector_arrow(fig, axes[k, 1], config.X, config.Y, output_tmp,
                                                'Predicted E', max_value=target_max)
            plot_ax_vector_arrow(fig, axes[k, 2], config.X, config.Y, target_tmp,
                                                'Target E', max_value=target_max)
            plot_ax_scalar(fig, axes[k, 3], config.X, config.Y, np.abs(norm_target_tmp - norm_output_tmp),
                                'Residual', cmap='Blues')
        else:
            plot_ax_scalar(fig, axes[k, 0], config.X, config.Y, data_tmp, '')
            # Same scale for output and target
            plot_ax_vector_arrow(fig, axes[k, 1], config.X, config.Y, output_tmp,
                                                '', max_value=target_max)
            plot_ax_vector_arrow(fig, axes[k, 2], config.X, config.Y, target_tmp,
                                                '', max_value=target_max)
            plot_ax_scalar(fig, axes[k, 3], config.X, config.Y, np.abs(norm_target_tmp - norm_output_tmp),
                                '', cmap='Blues')

    return fig


def plot_distrib(output, target, epoch, batch_idx):
    """ Plot distribution (ie. flattened target vs. output) """
    output_np = output[batch_idx, 0].detach().flatten().cpu().numpy()
    target_np = target[batch_idx, 0].detach().flatten().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title(f'Output distribution at epoch {epoch} for batch {batch_idx}')

    ax.plot(target_np, output_np, '.', markersize=1.5)
    ax.set_xlabel('target')
    ax.set_ylabel('output')

    return fig

def plot_scales(output, target, data, epoch, batch_idx, config):
    """ Matplotlib plots. """
    # Detach tensors and send them to cpu as numpy
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    data_np = data.detach().cpu().numpy()

    # Lots of plots
    fig, axes = plt.subplots(figsize=(20, 16), nrows=4, ncols=6)
    fig.suptitle('Epoch {} batch {}'.format(epoch, batch_idx), fontsize=16, y=0.95)

    for k in range(4):  # First 4 items of the batch
        if config.channels == 3:
            data_tmp = data_np[batch_idx + k, 2]
        elif config.channels == 2:
            data_tmp = data_np[batch_idx + k, 1]
        else:
            data_tmp = data_np[batch_idx + k, 0]
        output_tmp = output_np[batch_idx + k, 0]
        output_big = output_np[batch_idx + k, 1]
        output_medium = output_np[batch_idx + k, 2]
        output_small = output_np[batch_idx + k, 3]
        target_tmp = target_np[batch_idx + k, 0]
        # Same scale for output and target
        target_max = round_up(np.max(np.abs(target_tmp)), decimals=1)
        plot_ax_scalar(fig, axes[k, 0], config.X, config.Y, data_tmp, 'rhs')
        plot_ax_scalar(fig, axes[k, 1], config.X, config.Y, output_tmp, 'predicted potential', max_value=target_max)
        plot_ax_scalar(fig, axes[k, 2], config.X, config.Y, output_big, 'predicted Big Scale', max_value=target_max)
        plot_ax_scalar(fig, axes[k, 3], config.X, config.Y, output_medium, 'predicted Medium Scale', max_value=target_max)
        plot_ax_scalar(fig, axes[k, 4], config.X, config.Y, output_small, 'predicted Small Scale', max_value=target_max)
        plot_ax_scalar(fig, axes[k, 5], config.X, config.Y, target_tmp, 'target potential', max_value=target_max)
    return fig
