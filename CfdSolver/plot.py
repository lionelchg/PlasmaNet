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

def round_up(n, decimals=0): 
    multiplier = 10 ** decimals 
    return np.ceil(n * multiplier) / multiplier

def plot_ax_scalar(fig, ax, X, Y, field, title, cmap_scale=None, cmap='RdBu', geom='xr', field_ticks=[1e14, 1e17, 1e20]):
    fraction_cbar = 0.1
    aspect = 1.7 * np.max(Y) / fraction_cbar / np.max(X)
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
        max_value = round_up(np.max(np.abs(field)), decimals=1)
        levels = np.linspace(- max_value, max_value, 101)
        cs1 = ax.contourf(X, Y, field, levels, cmap=cmap)
        if geom == 'xr':
            ax.contourf(X, - Y, field, levels, cmap=cmap)
        fig.colorbar(cs1, ax=ax, pad=0.05, fraction=fraction_cbar, aspect=aspect)

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

def plot_scalar(X, Y, u, res, dtsum, number, fig_dir, geom='xy'):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    plot_ax_scalar(fig, axes[0], X, Y, u, "Scalar", geom=geom)
    plot_ax_scalar(fig, axes[1], X, Y, res, "Residual", geom=geom)
    plt.tight_layout()
    plt.figtext(0.85, 0.07, '$t =$%.2e s' % dtsum, fontsize=12)
    plt.savefig(fig_dir + 'instant_%04d' % number, bbox_inches='tight')

    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    plot_ax_scalar_1D(fig, axes[0], X, [0, 0.05, 0.1], u, "Scalar")
    plot_ax_scalar_1D(fig, axes[1], X, [0, 0.05, 0.1], res, "Residual")
    plt.tight_layout()
    plt.figtext(0.85, 0.07, '$t =$%.2e s' % dtsum, fontsize=12)
    plt.savefig(fig_dir + 'instant_1D_%04d' % number, bbox_inches='tight')

def plot_streamer(X, Y, ne, rese, nionp, resp, nn, resn, dtsum, number, fig_dir):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
    plot_ax_scalar(fig, axes[0][0], X, Y, ne, "$n_e$", cmap_scale='log')
    plot_ax_scalar(fig, axes[0][1], X, Y, rese, "$r_e$")
    plot_ax_scalar(fig, axes[1][0], X, Y, nionp, "$n_p$", cmap_scale='log')
    plot_ax_scalar(fig, axes[1][1], X, Y, resp, "$r_p$")
    plot_ax_scalar(fig, axes[2][0], X, Y, nn, "$n_n$", cmap_scale='log')
    plot_ax_scalar(fig, axes[2][1], X, Y, resn, "$r_n$")
    plt.tight_layout()
    plt.figtext(0.85, 0.07, '$t =$%.2e s' % dtsum, fontsize=12)
    plt.savefig(fig_dir + 'dens_%04d' % number, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
    plot_ax_scalar_1D(fig, axes[0][0], X, [0, 0.05, 0.1], ne, "$n_e$")
    plot_ax_scalar_1D(fig, axes[0][1], X, [0, 0.05, 0.1], rese, "$r_e$")
    plot_ax_scalar_1D(fig, axes[1][0], X, [0, 0.05, 0.1], nionp, "$n_p$")
    plot_ax_scalar_1D(fig, axes[1][1], X, [0, 0.05, 0.1], resp, "$r_p$")
    plot_ax_scalar_1D(fig, axes[2][0], X, [0, 0.05, 0.1], nn, "$n_n$")
    plot_ax_scalar_1D(fig, axes[2][1], X, [0, 0.05, 0.1], resn, "$r_n$")
    plt.tight_layout()
    plt.figtext(0.85, 0.07, '$t =$%.2e s' % dtsum, fontsize=12)
    plt.savefig(fig_dir + 'dens_cut_%04d' % number, bbox_inches='tight')
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