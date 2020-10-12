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
        plot_ax_scalar(fig, axes[k, 0], config.X, config.Y, data_tmp, 'rhs')
        plot_ax_scalar(fig, axes[k, 1], config.X, config.Y, output_tmp, 'predicted potential', max_value=target_max)
        plot_ax_scalar(fig, axes[k, 2], config.X, config.Y, target_tmp, 'target potential', max_value=target_max)
        plot_ax_scalar(fig, axes[k, 3], config.X, config.Y, np.abs(target_tmp - output_tmp), 'residual',
                       colormap='Blues')

    return fig

def plot_batch_2(output, target, data, epoch, batch_idx, config):
    """ Matplotlib plots. """
    # Detach tensors and send them to cpu as numpy
    data_np = data.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()


    # Lots of plots
    fig, axes = plt.subplots(figsize=(20, 16), nrows=4, ncols=4)
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
        plot_ax_scalar(fig, axes[k, 0], config.X, config.Y, data_tmp, 'rhs')
        plot_ax_scalar(fig, axes[k, 1], config.X, config.Y, output_tmp, 'predicted potential')
        plot_ax_scalar(fig, axes[k, 2], config.X, config.Y, target_tmp, 'target potential')
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

def plot_ax_vector_arrow(fig, ax, X, Y, vector_field, name, colormap='Blues'):
    norm_field = np.sqrt(vector_field[0]**2 + vector_field[1]**2)
    arrow_step = 10
    CS = ax.contourf(X, Y, norm_field, 100, cmap=colormap)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.1, ax=ax, aspect=7)
    q = ax.quiver(X[::arrow_step, ::arrow_step], Y[::arrow_step, ::arrow_step], 
                vector_field[0, ::arrow_step, ::arrow_step], vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
    ax.set_title(name)
    ax.set_aspect('equal')

def plot_set_2D(X, Y, physical_rhs, test_pot, target_pot, E_test, E_target, figtitle, figname):
    """ Matplotlib plots. """
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(21, 10))
    plot_ax_scalar(fig, axes[0, 0], X, Y, physical_rhs, r'$\rho / \epsilon_0$')
    plot_ax_scalar(fig, axes[0, 1], X, Y, test_pot, r'$\phi$ test')
    plot_ax_scalar(fig, axes[0, 2], X, Y, target_pot, r'$\phi$ target')
    plot_ax_scalar(fig, axes[0, 3], X, Y, np.abs(target_pot - test_pot), 'Residual',
                   colormap='Blues')
    plot_ax_vector_arrow(fig, axes[1, 0], X, Y, E_test, r'$\mathbf{E}$ test')
    plot_ax_vector_arrow(fig, axes[1, 1], X, Y, E_target, r'$\mathbf{E}$ target')
    normE_target = np.sqrt(E_target[0]**2 + E_target[1]**2)
    normE_test = np.sqrt(E_test[0]**2 + E_test[1]**2)
    angle = np.arccos((E_test[0]*E_target[0] + E_test[1]*E_target[1]) / normE_target / normE_test)
    plot_ax_scalar(fig, axes[1, 2], X, Y, np.abs(normE_target - normE_test), 'Norm residual',
                   colormap='Blues')
    plot_ax_scalar(fig, axes[1, 3], X, Y, angle * 180 / np.pi, 'Angle',
                   colormap='Blues')
    plt.suptitle(figtitle)
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')


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
