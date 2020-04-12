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

def plot_set_1D(x, pot_target, E_field_norm_target, lapl_target, pot, E_field_norm, lapl_pot, n_points, figtitle, figname):
    # 1D plot
    n_middle = int(n_points / 2)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))
    axes[0].plot(x, pot_target[n_middle, :], label=r'Target $\phi$')
    axes[1].plot(x, E_field_norm_target[n_middle, :], label=r'Target $\mathbf{E}$')
    axes[2].plot(x, lapl_target[n_middle, :], label=r'Target $\nabla^2 \phi$')

    axes[0].plot(x, pot[n_middle, :])
    axes[1].plot(x, E_field_norm[n_middle, :])
    axes[2].plot(x, lapl_pot[n_middle, :])

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plt.suptitle(figtitle)
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')

def plot_set_2D(X, Y, physical_rhs, test_pot, target_pot, E_test, E_target, figtitle, figname):
    """ Matplotlib plots. """
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(25, 12))
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


def plot_ax_scalar(fig, ax, X, Y, field, title, colormap='RdBu'):
    if colormap == 'RdBu':
        max_value = round_up(np.max(np.abs(field)), decimals=1)
        levels = np.linspace(- max_value, max_value, 101)
    else:
        levels = 101
    cs1 = ax.contourf(X, Y, field, levels, cmap=colormap)
    fig.colorbar(cs1, ax=ax, pad=0.05, fraction=0.1, aspect=7)
    ax.set_aspect("equal")
    ax.set_title(title)


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier


def plot_ax_vector_arrow(fig, ax, X, Y, vector_field, name, colormap='Blues'):
    norm_field = np.sqrt(vector_field[0]**2 + vector_field[1]**2)
    arrow_step = 10
    CS = ax.contourf(X, Y, norm_field, 100, cmap=colormap)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.1, ax=ax, aspect=7)
    q = ax.quiver(X[::arrow_step, ::arrow_step], Y[::arrow_step, ::arrow_step], 
                vector_field[0, ::arrow_step, ::arrow_step], vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
    ax.set_title(name)
    ax.set_aspect('equal')
