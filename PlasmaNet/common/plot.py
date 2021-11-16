import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib as mpl

from .operators_numpy import lapl, grad

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['contour.negative_linestyle'] = 'solid'
default_cmap = 'RdBu'  # 'RdBu'


def round_up(number: float, decimals:int=1) -> float:
    """ Rounding of a number

    :param number: A number to round
    :type number: float
    :param decimals: Number of decimals defaults to 1
    :type decimals: int, optional
    :return: The rounded number
    :rtype: float
    """

    if number != 0.0:
        power = int(np.log10(number))
        digit = number / 10**power * 10**decimals
        return np.ceil(digit) * 10**(power - decimals)
    else:
        return 1.0


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


def plot_set_2D(X, Y, physical_rhs, pot, E, figtitle, figname, no_rhs=False, geom='xy'):
    """ 2D Matplotlib plots. """
    if no_rhs:
        fig, axes = plt.subplots(ncols=2, figsize=(11, 4))
        plot_ax_scalar(fig, axes[0], X, Y, pot, r'$\phi$', geom=geom)
        plot_ax_vector_arrow(fig, axes[1], X, Y, E, r'$\mathbf{E}$', geom=geom)
    else:
        if geom == 'xy':
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(221)
            plot_ax_scalar(fig, ax, X, Y, physical_rhs, r'$\rho / \epsilon_0$', geom=geom)
            ax = fig.add_subplot(222)
            plot_ax_scalar(fig, ax, X, Y, pot, r'$\phi$', geom=geom)
            ax = fig.add_subplot(212)
            plot_ax_vector_arrow(fig, ax, X, Y, E, r'$\mathbf{E}$', geom=geom)
        else:
            fig, axes = plt.subplots(ncols=3, figsize=(14, 4))
            plot_ax_scalar(fig, axes[0], X, Y, physical_rhs, r'$\rho / \epsilon_0$')
            plot_ax_scalar(fig, axes[1], X, Y, pot, r'$\phi$')
            plot_ax_vector_arrow(fig, axes[2], X, Y, E, r'$\mathbf{E}$')
    plt.suptitle(figtitle)
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')
    plt.close()


def plot_ax_scalar(fig, ax, X, Y, field, title, cmap_scale=None, cmap='RdBu',
        geom='xy', field_ticks=None, max_value=None, cbar=True, contour=True):
    """ Plot a 2D field on mesh X and Y with contourf and contour. Automatic
    handling of maximum values for the colorbar with an up rounding to a certain
    number of decimals. """
    # If not specified find the maximum value and round this up to 1 decimal
    if max_value is None:
        max_value = round_up(np.max(np.abs(field)), decimals=1)
    else:
        max_value = round_up(max_value, decimals=1)

    # Depending on the scale (log is typically for streamers) the treatment
    # is not the same
    if cmap_scale == 'log':
        cmap = 'Blues'
        if field_ticks is None:
            field_ticks = [10**int(np.log10(max_value)) / 10**(3 - tmp_pow) for tmp_pow in range(5)]
            pows = np.log10(np.array(field_ticks)).astype(int)
            levels = np.logspace(pows[0], pows[-1], 100, endpoint=True)
        else:
            pows = np.log10(np.array(field_ticks)).astype(int)
            levels = np.logspace(pows[0], pows[-1], 100, endpoint=True)
        field = np.maximum(field, field_ticks[0])
        field = np.minimum(field, field_ticks[-1])
        cs1 = ax.contourf(X, Y, field, levels, cmap=cmap, norm=LogNorm())
        if geom == 'xr':
            ax.contourf(X, - Y, field, levels, cmap=cmap, norm=LogNorm())
        if contour:
            ax.contour(X, Y, field, levels=field_ticks[1:-1], colors='k', linewidths=0.9)
            if geom == 'xr':
                ax.contour(X, - Y, field, levels=field_ticks[1:-1], colors='k', linewidths=0.9)
    else:
        if cmap == 'Blues':
            field_ticks = np.linspace(0, max_value, 5)
            levels = np.linspace(0, max_value, 101)
        else:
            field_ticks = np.linspace(-max_value, max_value, 5)
            levels = np.linspace(-max_value, max_value, 101)
        cs1 = ax.contourf(X, Y, field, levels, cmap=cmap)
        if geom == 'xr':
            ax.contourf(X, - Y, field, levels, cmap=cmap)
        if contour:
            if cmap == 'Blues':
                clevels = np.array([0.2, 0.5, 0.8]) * np.max(np.abs(field))
            else:
                clevels = np.array([- 0.8, - 0.2, 0.2, 0.8]) * np.max(np.abs(field))
            ax.contour(X, Y, field, levels=clevels, colors='k', linewidths=0.9)
            if geom == 'xr':
                ax.contour(X, -Y, field, levels=clevels, colors='k', linewidths=0.9)

    # Put colorbar if specified
    xmax, ymax = np.max(X), np.max(Y)
    if cbar:
        # Adjust the size of the colorbar
        xmax, ymax = np.max(X), np.max(Y)
        fraction_cbar = 0.1
        if geom == 'xr':
            aspect = 1.7 * ymax / fraction_cbar / xmax
        else:
            aspect = 0.85 * ymax / fraction_cbar / xmax
        # Set the colorbar in scientific notation
        # sfmt = mpl.ticker.ScalarFormatter(useMathText=True)
        # sfmt = mpl.ticker.ScalarFormatter()
        # sfmt.set_powerlimits((0, 0))
        # fig.colorbar(cs1, ax=ax, pad=0.05, fraction=fraction_cbar, aspect=aspect,
        #     ticks=field_ticks, format=sfmt)
        fig.colorbar(cs1, ax=ax, pad=0.05, fraction=fraction_cbar, aspect=aspect,
            ticks=field_ticks)

    if geom == 'xr':
        ax.set_yticks([-ymax, -ymax / 2, 0, ymax / 2, ymax])

    # Apply same formatting to x and y axis with scientific notation
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.set_aspect("equal")
    ax.set_title(title)


def plot_ax_scalar_1D(fig, ax, X, list_cut, field, title, yscale='linear', ylim=None):
    x = X[0, :]
    n_points = len(X[:, 0])

    for cut_pos in list_cut:
        n = int(cut_pos * (n_points - 1))
        ax.plot(x, field[n, :], label='$\hat{y}$ = %.2f' % cut_pos)
    ax.legend(loc='upper right')
    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)
    ax.set_title(title)
    scilimx = int(np.log10(np.max(x)))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(scilimx, scilimx))


def plot_ax_vector_arrow(fig, ax, X, Y, vector_field, name, colormap='Blues',
                            geom='xy', max_value=None, cbar=True):
    norm_field = np.sqrt(vector_field[0]**2 + vector_field[1]**2)
    if geom == 'xy':
        arrow_step = int(len(X[:, 0]) / 10)
    elif geom == 'xr':
        arrow_step = int(len(X[:, 0]) / 5)

    if max_value is None:
        max_value = round_up(np.max(np.abs(norm_field)), decimals=1)
    else:
        max_value = round_up(max_value, decimals=1)

    levels = np.linspace(0, max_value, 101)
    CS = ax.contourf(X, Y, norm_field, levels, cmap=colormap)
    fraction_cbar = 0.1

    if geom == 'xr':
        ax.contourf(X, - Y, norm_field, levels, cmap=colormap)
        aspect = 1.7 * np.max(Y) / fraction_cbar / np.max(X)
    else:
        aspect = 0.85 * np.max(Y) / fraction_cbar / np.max(X)

    if cbar:
        # Set the colorbar in scientific notation
        sfmt = mpl.ticker.ScalarFormatter(useMathText=True)
        sfmt.set_powerlimits((0, 0))
        fig.colorbar(CS, pad=0.05, fraction=fraction_cbar, ax=ax, aspect=aspect,
            ticks=np.linspace(0, max_value, 5), format=sfmt)
    ax.quiver(X[::arrow_step, ::arrow_step], Y[::arrow_step, ::arrow_step],
                vector_field[0, ::arrow_step, ::arrow_step], vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')

    if geom == 'xr':
        ax.quiver(X[::arrow_step, ::arrow_step], - Y[::arrow_step, ::arrow_step],
            vector_field[0, ::arrow_step, ::arrow_step], - vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')

    ax.set_title(name)
    ax.set_aspect('equal')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))


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
    plot_ax_scalar(fig, axes[2, 0], X, Y, - lapl_trial, 'Charge density')
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


def plot_modes(ax, N, M, coeffs, title, cmap_str='Blues'):
    """ Plot of the different modes of a 2D Fourier expansion """
    # ax.plot_surface(N, M, coeffs, alpha=0.7)
    N, M, coeffs = N.reshape(-1), M.reshape(-1), coeffs.reshape(-1)

    offset = coeffs + np.abs(coeffs.min())
    fracs = offset.astype(float) / offset.max()
    norm = mpl.colors.Normalize(fracs.min(), fracs.max())
    cmap = getattr(mpl.cm, cmap_str)
    colors = cmap(norm(fracs))

    ax.bar3d(N, M, np.zeros_like(N), np.ones_like(N), np.ones_like(M), coeffs, alpha=0.8, color=colors)
    ax.set_zlabel('Amplitude')
    ax.set_ylabel('M')
    ax.set_xlabel('N')
    ax.set_title(title)
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
    ax.view_init(elev=20, azim=35)
