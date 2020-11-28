########################################################################################################################
#                                                                                                                      #
#                                                    Plot functions                                                    #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def round_up(n, decimals=1):
    power = int(np.log10(n))
    digit = n / 10**power * 10**decimals
    return np.ceil(digit) * 10**(power - decimals)


def plot_ax_scalar(fig, ax, X, Y, field, title, cmap_scale=None, cmap='RdBu', 
        geom='xr', field_ticks=None, max_value=None):
    # Avoid mutable defaults value
    if field_ticks is None:
        field_ticks = [1e14, 1e17, 1e20]
    xmax, ymax = np.max(X), np.max(Y)
    fraction_cbar = 0.1
    aspect = 1.7 * ymax / fraction_cbar / xmax
    if cmap_scale == 'log':
        cmap = 'Blues'
        pows = np.log10(np.array(field_ticks)).astype(int)
        levels = np.logspace(pows[0], pows[-1], 100, endpoint=True)
        field = np.maximum(field, field_ticks[0])
        field = np.minimum(field, field_ticks[-1])
        cs1 = ax.contourf(X, Y, field, levels, cmap=cmap, norm=LogNorm())
        if geom == 'xr':
            ax.contourf(X, - Y, field, levels, cmap=cmap, norm=LogNorm())
        fig.colorbar(cs1, ax=ax, pad=0.05, fraction=fraction_cbar, aspect=aspect, ticks=field_ticks)
    else:
        if max_value is None:
            max_value = round_up(np.max(np.abs(field)), decimals=1)
        levels = np.linspace(- max_value, max_value, 101)
        cs1 = ax.contourf(X, Y, field, levels, cmap=cmap)
        if geom == 'xr':
            ax.contourf(X, - Y, field, levels, cmap=cmap)
        fig.colorbar(cs1, ax=ax, pad=0.05, fraction=fraction_cbar, aspect=aspect)

    if geom == 'xr':
        ax.set_yticks([-ymax, -ymax / 2, 0, ymax / 2, ymax])

    scilimx = int(np.log10(xmax) - 1)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(scilimx, scilimx))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(scilimx, scilimx))

    ax.set_aspect("equal")
    ax.set_title(title)


def plot_ax_scalar_1D(fig, ax, X, list_cut, field, title, yscale='linear', ylim=None):
    x = X[0, :]
    n_points = len(X[:, 0])

    for cut_pos in list_cut:
        n = int(cut_pos * (n_points - 1))
        ax.plot(x, field[n, :], label='y = %.2f ymax' % cut_pos)
    ax.legend()
    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)

    ax.set_title(title)


def plot_streamer(X, Y, nd, resnd, dtsum, figname):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
    plot_ax_scalar(fig, axes[0][0], X, Y, nd[0], "$n_e$", cmap_scale='log')
    plot_ax_scalar(fig, axes[0][1], X, Y, resnd[0], "$r_e$")
    plot_ax_scalar(fig, axes[1][0], X, Y, nd[1], "$n_p$", cmap_scale='log')
    plot_ax_scalar(fig, axes[1][1], X, Y, resnd[1], "$r_p$")
    plot_ax_scalar(fig, axes[2][0], X, Y, nd[2], "$n_n$", cmap_scale='log')
    plot_ax_scalar(fig, axes[2][1], X, Y, resnd[2], "$r_n$")
    plt.suptitle(f'$t$ = {dtsum:.2e} s')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(figname, bbox_inches='tight')
    plt.close()


def plot_streamer_1D(X, Y, nd, resnd, dtsum, cut_pos, figname):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
    plot_ax_scalar_1D(fig, axes[0][0], X, cut_pos, nd[0], "$n_e$", yscale='log')
    plot_ax_scalar_1D(fig, axes[0][1], X, cut_pos, resnd[0], "$r_e$")
    plot_ax_scalar_1D(fig, axes[1][0], X, cut_pos, nd[1], "$n_p$", yscale='log')
    plot_ax_scalar_1D(fig, axes[1][1], X, cut_pos, resnd[1], "$r_p$")
    plot_ax_scalar_1D(fig, axes[2][0], X, cut_pos, nd[2], "$n_n$", yscale='log')
    plot_ax_scalar_1D(fig, axes[2][1], X, cut_pos, resnd[2], "$r_n$")
    plt.suptitle(f'$t$ = {dtsum:.2e} s')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(figname, bbox_inches='tight')
    plt.close()


def plot_global(gstreamer, xrange, figname):
    """ Global quantities (position of negative streamer, 
    positive streamer and energy of discharge) """
    time = gstreamer[:, 0] / 1e-9
    gstreamer[:, 1:3] = gstreamer[:, :2] / 1e-3
    gstreamer[:, 3] = gstreamer[:, 3] / 1e-6
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    axes[0].plot(time, gstreamer[:, 1], label='Negative streamer')
    axes[0].plot(time, gstreamer[:, 2], label='Positive streamer')
    axes[0].set_ylabel('$x$ [mm]')
    axes[0].set_xlabel('$t$ [ns]')
    axes[0].set_ylim(np.array(xrange) / 1e-3)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time, gstreamer[:, 3])
    axes[1].set_xlabel('$t$ [ns]')
    axes[1].set_ylabel('E [$\mu$J]')
    axes[1].grid(True)

    plt.savefig(figname, bbox_inches='tight')


def plot_ax_vector_arrow(fig, ax, X, Y, vector_field, name, colormap='Blues', axi=False, max_value=None):
    norm_field = np.sqrt(vector_field[0]**2 + vector_field[1]**2)
    arrow_step = 10
    if max_value is None:
        max_value = round_up(np.max(np.abs(norm_field)), decimals=1)
    levels = np.linspace(0, max_value, 101)
    CS = ax.contourf(X, Y, norm_field, levels, cmap=colormap)
    fraction_cbar = 0.1
    if axi:
        ax.contourf(X, - Y, norm_field, levels, cmap=colormap)
        aspect = 1.7 * np.max(Y) / fraction_cbar / np.max(X)
    else:
        aspect = 0.85 * np.max(Y) / fraction_cbar / np.max(X)
    fig.colorbar(CS, pad=0.05, fraction=fraction_cbar, ax=ax, aspect=aspect)
    ax.quiver(X[::arrow_step, ::arrow_step], Y[::arrow_step, ::arrow_step], 
                vector_field[0, ::arrow_step, ::arrow_step], vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
    if axi:
        ax.quiver(X[::arrow_step, ::arrow_step], - Y[::arrow_step, ::arrow_step], 
            vector_field[0, ::arrow_step, ::arrow_step], - vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
    ax.set_title(name)
    ax.set_aspect('equal')