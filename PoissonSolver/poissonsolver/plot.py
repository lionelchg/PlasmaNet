import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from poissonsolver.operators import lapl, grad

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['contour.negative_linestyle'] = 'solid'
default_cmap = 'RdBu'  # 'RdBu'


def round_up(n, decimals=1):
    power = int(np.log10(n))
    digit = n / 10**power * 10**decimals
    return np.ceil(digit) * 10**(power - decimals)


def plot_ax_set_1D(axes, x, pot, E_field_norm, lapl_pot, n_points, M, direction='x'):
    # 1D plot
    n_middle = int(n_points / 2)
    axes[0].plot(x, - lapl_pot[n_middle, :], '.', label='M = %d' % M)
    axes[0].legend()
    axes[1].plot(x, pot[n_middle, :])
    axes[1].set_title(r'$\phi$')
    axes[2].plot(x, E_field_norm[n_middle, :])
    axes[2].set_title(r'$\mathbf{E}$')


def plot_set_1D(x, physical_rhs, pot, E_field_norm, lapl_pot, n_points, figtitle, figname, no_rhs=False, direction='y'):
    """ 1D Matplotlib plots with cuts accross the y axis. """
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
        axes[1].set_ylim([0, 1.1 * np.max(E_field_norm)])
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
        axes[2].set_ylim([0, 1.2 * np.max(E_field_norm)])

    axes[0].grid()
    axes[1].grid()
    axes[0].legend()
    axes[1].legend()
    plt.suptitle(figtitle, y=1)
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')
    plt.close()


def plot_set_2D(X, Y, physical_rhs, pot, E, figtitle, figname, no_rhs=False, axi=False):
    """ 2D Matplotlib plots. """
    if no_rhs:
        fig, axes = plt.subplots(ncols=2, figsize=(11, 4))
        plot_ax_scalar(fig, axes[0], X, Y, pot, r'$\phi$', axi=axi)
        plot_ax_vector_arrow(fig, axes[1], X, Y, E, r'$\mathbf{E}$', axi=axi)
    else:
        if axi:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(221)
            plot_ax_scalar(fig, ax, X, Y, physical_rhs, r'$\rho / \epsilon_0$', axi=axi)
            ax = fig.add_subplot(222)
            plot_ax_scalar(fig, ax, X, Y, pot, r'$\phi$', axi=axi)
            ax = fig.add_subplot(212)
            plot_ax_vector_arrow(fig, ax, X, Y, E, r'$\mathbf{E}$', axi=axi)
        else:
            fig, axes = plt.subplots(ncols=3, figsize=(14, 4))
            plot_ax_scalar(fig, axes[0], X, Y, physical_rhs, r'$\rho / \epsilon_0$')
            plot_ax_scalar(fig, axes[1], X, Y, pot, r'$\phi$')
            plot_ax_vector_arrow(fig, axes[2], X, Y, E, r'$\mathbf{E}$')
    plt.suptitle(figtitle)
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')
    plt.close()


def plot_ax_scalar(fig, ax, X, Y, field, title, colormap='RdBu', axi=False):
    xmax, ymax = np.max(X[:]), np.max(Y[:])
    if colormap == 'RdBu':
        max_value = round_up(np.max(np.abs(field)), decimals=1)
        levels = np.linspace(- max_value, max_value, 101)
    else:
        levels = 101
    cs1 = ax.contourf(X, Y, field, levels, cmap=colormap)
    fraction_cbar = 0.1
    if axi: 
        ax.contourf(X, - Y, field, levels, cmap=colormap)
        aspect = 1.7 * ymax / fraction_cbar / xmax
        ymax = np.max(Y[:])
        ax.set_yticks([-ymax, -ymax / 2, 0, ymax / 2, ymax])
    else:
        aspect = 0.85 * ymax / fraction_cbar / xmax
    fig.colorbar(cs1, ax=ax, pad=0.05, fraction=fraction_cbar, aspect=aspect)

    scilimx = int(np.log10(xmax) - 1)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(scilimx, scilimx))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(scilimx, scilimx))

    ax.set_aspect("equal")
    ax.set_title(title)


def plot_ax_vector_arrow(fig, ax, X, Y, vector_field, name, colormap='Blues', axi=False):
    xmax, ymax = np.max(X[:]), np.max(Y[:])
    norm_field = np.sqrt(vector_field[0]**2 + vector_field[1]**2)
    arrow_step = 20
    levels = np.linspace(0, np.max(norm_field), 101)
    CS = ax.contourf(X, Y, norm_field, levels, cmap=colormap)
    fraction_cbar = 0.1
    if axi:
        ax.contourf(X, - Y, norm_field, levels, cmap=colormap)
        aspect = 1.7 * ymax / fraction_cbar / xmax
    else:
        aspect = 0.85 * ymax / fraction_cbar / xmax
    cbar = fig.colorbar(CS, pad=0.05, fraction=fraction_cbar, ax=ax, aspect=aspect)
    q = ax.quiver(X[::arrow_step, ::arrow_step], Y[::arrow_step, ::arrow_step], 
                vector_field[0, ::arrow_step, ::arrow_step], vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
    if axi:
        q = ax.quiver(X[::arrow_step, ::arrow_step], - Y[::arrow_step, ::arrow_step], 
            vector_field[0, ::arrow_step, ::arrow_step], vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
    ax.set_title(name)
    xmax, ymax = np.max(X[:]), np.max(Y[:])
    scilimx = int(np.log10(xmax) - 1)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(scilimx, scilimx))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(scilimx, scilimx))
    ax.set_aspect('equal')


def plot_potential(X, Y, dx, dy, potential, nx, ny, figname, figtitle=None, r=None):
    # 1D vector
    x = X[0, :]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 14))

    plot_ax_scalar(fig, axes[0][0], X, Y, potential, 'Potential')
    plot_ax_trial_1D(axes[0][1], x, potential, ny, '1D cuts')

    E = - grad(potential, dx, dy, nx, ny)
    normE = np.sqrt(E[0]**2 + E[1]**2)
    plot_ax_vector_arrow(fig, axes[1][0], X, Y, E, 'Electric field')
    plot_ax_trial_1D(axes[1][1], x, normE, ny, '1D cuts', ylim=[0.99 * np.min(normE), 1.01 * np.max(normE)])

    if r is not None:
        lapl_trial = lapl(potential, dx, dy, nx, ny, r=r)
    else:
        lapl_trial = lapl(potential, dx, dy, nx, ny)
    plot_ax_scalar(fig, axes[2, 0], X, Y, - lapl_trial, '- Laplacian')
    plot_ax_trial_1D(axes[2][1], x, -  lapl_trial, ny, '1D cuts')

    if figtitle is not None:
        plt.suptitle(figtitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(figname, bbox_inches='tight')
    plt.close()


def plot_ax_trial_1D(ax, x, function, n_points, title, direction='y', ylim=None):
    list_cut = [0, 0.25, 0.5, 0.75, 1.0]
    for cut_pos in list_cut:
        n = int(cut_pos * (n_points - 1))
        ax.plot(x, function[n, :], label='%s = %.2f %smax' % (direction, cut_pos, direction))
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_lapl_rhs(X, Y, dx, dy, potential, physical_rhs, nx, ny, figname, figtitle=None, r=None):
    """ Compares the laplacian of the potential against the real rhs """
    x = X[0, :]

    fig = plt.figure(figsize=(8, 10))

    ax = fig.add_subplot(221)
    if r is not None:
        lapl_trial = lapl(potential, dx, dy, nx, ny, r=r)
    else:
        lapl_trial = lapl(potential, dx, dy, nx, ny)
    plot_ax_scalar(fig, ax, X, Y, - lapl_trial, '- Laplacian')

    ax = fig.add_subplot(222)
    plot_ax_scalar(fig, ax, X, Y, physical_rhs, 'RHS')

    ax = fig.add_subplot(212)
    plot_ax_trial_1D(ax, x, -  lapl_trial, ny, '1D cuts', direction='y lapl')
    plot_ax_trial_1D(ax, x, physical_rhs, ny, '1D cuts', direction='y rhs')

    if figtitle is not None:
        plt.suptitle(figtitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(figname, bbox_inches='tight')
    plt.close()
