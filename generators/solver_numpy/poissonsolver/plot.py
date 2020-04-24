import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from poissonsolver.operators import lapl, grad, derivative

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['contour.negative_linestyle'] = 'solid'
default_cmap = 'RdBu'  # 'RdBu'

def round_up(n, decimals=0): 
    multiplier = 10 ** decimals 
    return np.ceil(n * multiplier) / multiplier


def plot_ax_set_1D(axes, x, pot, E_field_norm, lapl_pot, n_points, M, direction='x'):
    # 1D plot
    n_middle = int(n_points / 2)
    axes[0].plot(x, - lapl_pot[n_middle, :], '.', label='M = %d' % M)
    axes[0].legend()
    axes[1].plot(x, pot[n_middle, :])
    axes[1].set_title(r'$\phi$')
    axes[2].plot(x, E_field_norm[n_middle, :])
    axes[2].set_title(r'$\mathbf{E}$')


def plot_set_1D(x, physical_rhs, pot, E_field_norm, lapl_pot, n_points, figtitle, figname, no_rhs=False, direction='x'):
    # 1D plot
    n_middle = int(n_points / 2)
    if no_rhs:
        list_cut = [0, 0.25, 0.5, 0.75, 1]
        fig, axes = plt.subplots(ncols=2, figsize=(10, 7))
        for cut_pos in list_cut:
            n = int(cut_pos * (n_points - 1))
            axes[0].plot(x, pot[n, :], label='%s = %.2f %smax' % (direction, cut_pos, direction))
            axes[1].plot(x, E_field_norm[n, :], label='%s = %.2f %smax' % (direction, cut_pos, direction))
        axes[0].set_title(r'$\phi$')
        axes[1].set_title(r'$\mathbf{E}$')
    else:
        list_cut = [0, 0.25, 0.5]
        fig, axes = plt.subplots(ncols=3, figsize=(15, 7))
        for cut_pos in list_cut:        
            n = int(cut_pos * (n_points - 1))
            axes[1].plot(x, pot[n, :], label='%s = %.2f %smax' % (direction, cut_pos, direction))
            axes[2].plot(x, E_field_norm[n, :], label='%s = %.2f %smax' % (direction, cut_pos, direction))

        axes[0].plot(x, - lapl_pot[n_middle, :], '.', label=r'$\nabla^2 \phi$')
        axes[0].plot(x, physical_rhs[n_middle, :], label=r'$\rho / \epsilon_0$')
        axes[1].set_title(r'$\phi$')
        axes[2].set_title(r'$\mathbf{E}$')
        axes[2].grid()
        axes[2].legend()

    axes[0].grid()
    axes[1].grid()
    axes[0].legend()
    axes[1].legend()
    plt.suptitle(figtitle, y=1)
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')

def plot_set_2D(X, Y, physical_rhs, pot, E, figtitle, figname, no_rhs=False):
    """ Matplotlib plots. """
    if no_rhs:
        fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
        plot_ax_scalar(fig, axes[0], X, Y, pot, r'$\phi$')
        plot_ax_vector_arrow(fig, axes[1], X, Y, E, r'$\mathbf{E}$')
    else:
        fig, axes = plt.subplots(ncols=3, figsize=(16, 5))
        plot_ax_scalar(fig, axes[0], X, Y, physical_rhs, r'$\rho / \epsilon_0$')
        plot_ax_scalar(fig, axes[1], X, Y, pot, r'$\phi$')
        plot_ax_vector_arrow(fig, axes[2], X, Y, E, r'$\mathbf{E}$')

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


def plot_ax_vector_arrow(fig, ax, X, Y, vector_field, name, colormap='Blues'):
    norm_field = np.sqrt(vector_field[0]**2 + vector_field[1]**2)
    arrow_step = 10
    CS = ax.contourf(X, Y, norm_field, 100, cmap=colormap)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.1, ax=ax, aspect=7)
    q = ax.quiver(X[::arrow_step, ::arrow_step], Y[::arrow_step, ::arrow_step], 
                vector_field[0, ::arrow_step, ::arrow_step], vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
    ax.set_title(name)
    ax.set_aspect('equal')

def plot_potential(X, Y, dx, dy, potential, n_points, figname, figtitle=None):
    # 1D vector
    x = X[0, :]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 14))

    plot_ax_scalar(fig, axes[0][0], X, Y, potential, 'Potential')
    plot_ax_trial_1D(axes[0][1], x, potential, n_points, '1D cuts')

    E = - grad(potential, dx, dy, n_points, n_points)
    normE = np.sqrt(E[0]**2 + E[1]**2)
    plot_ax_vector_arrow(fig, axes[1][0], X, Y, E, 'Electric field')
    plot_ax_trial_1D(axes[1][1], x, normE, n_points, '1D cuts')

    lapl_trial = lapl(potential, dx, dy, n_points, n_points)
    plot_ax_scalar(fig, axes[2, 0], X, Y, - lapl_trial, '- Laplacian')
    plot_ax_trial_1D(axes[2][1], x, -  lapl_trial, n_points, '1D cuts')

    if figtitle is not None:
        plt.suptitle(figtitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(figname, bbox_inches='tight')

def plot_ax_trial_1D(ax, x, function, n_points, title):
    direction = 'y'
    list_cut = [0, 0.25, 0.5]
    for cut_pos in list_cut:
        n = int(cut_pos * (n_points - 1))
        ax.plot(x, function[n, :], label='%s = %.2f %smax' % (direction, cut_pos, direction))
    ax.set_title(title)
    ax.legend()
    ax.grid(True)