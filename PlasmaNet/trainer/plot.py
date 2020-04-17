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

matplotlib.use('Agg')


def plot_batch(output, target, data, epoch, batch_idx, config):
    """ Matplotlib plots. """
    # Detach tensors and send them to cpu as numpy
    data_np = data.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()

    # Lots of plots
    fig, axes = plt.subplots(figsize=(20, 16), nrows=4, ncols=4)
    fig.suptitle('Epoch {} batch {}'.format(epoch, batch_idx), fontsize=16, y=0.95)

    for k in range(4):  # First 4 items of the batch
        data_tmp = data_np[batch_idx + k, 0]
        output_tmp = output_np[batch_idx + k, 0]
        target_tmp = target_np[batch_idx + k, 0]
        # Same scale for output and target
        target_max = round_up(np.max(np.abs(target_tmp)), decimals=1)
        plot_ax_scalar(fig, axes[k, 0], config.X, config.Y, data_tmp, 'rhs')
        plot_ax_scalar(fig, axes[k, 1], config.X, config.Y, output_tmp, 'predicted potential', max_value=target_max)
        plot_ax_scalar(fig, axes[k, 2], config.X, config.Y, target_tmp, 'target potential', max_value=target_max)
        plot_ax_scalar(fig, axes[k, 3], config.X, config.Y, np.abs(target_tmp - output_tmp), 'residual',
                       colormap='Blues')

    return fig


def plot_ax_scalar(fig, ax, X, Y, field, title, colormap='RdBu', max_value=None):
    if colormap == 'RdBu' and max_value is None:
        max_value = round_up(np.max(np.abs(field)), decimals=1)
        levels = np.linspace(- max_value, max_value, 101)
    elif colormap == 'RdBu':
        levels = np.linspace(- max_value, max_value, 101)
    else:
        levels = 101
    cs1 = ax.contourf(X, Y, field, levels, cmap=colormap)
    fig.colorbar(cs1, ax=ax)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier


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
