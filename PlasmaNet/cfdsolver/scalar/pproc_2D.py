import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path
import seaborn as sns
from matplotlib.colors import LogNorm

from PlasmaNet.cfdsolver.base.metric import Mesh
from PlasmaNet.common.plot import round_up

sns.set_context('notebook', font_scale=1.0)

def plot_ax_scalar(fig, ax, X, Y, field_up, field_down, title, cmap_scale=None, cmap='RdBu',
        geom='xy', field_ticks=None, max_value=None, cbar=True, contour=True):
    """ Plot a 2D field on mesh X and Y with contourf and contour. Automatic
    handling of maximum values for the colorbar with an up rounding to a certain
    number of decimals. """

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
        # Clipping up
        field_up = np.maximum(field_up, field_ticks[0])
        field_up = np.minimum(field_up, field_ticks[-1])

        # Clipping down
        field_down = np.maximum(field_down, field_ticks[0])
        field_down = np.minimum(field_down, field_ticks[-1])

        # Contourf
        cs1 = ax.contourf(X, Y, field_up, levels, cmap=cmap, norm=LogNorm())
        # Lower part
        ax.contourf(X, - Y, field_down, levels, cmap=cmap, norm=LogNorm())

        if contour:
            ax.contour(X, Y, field_up, levels=field_ticks[1:-1], colors='k', linewidths=0.9)
            # Lower part
            ax.contour(X, - Y, field_down, levels=field_ticks[1:-1], colors='k', linewidths=0.9)
    else:
        if cmap == 'Blues':
            field_ticks = np.linspace(0, max_value, 5)
            levels = np.linspace(0, max_value, 101)
        else:
            field_ticks = np.linspace(-max_value, max_value, 5)
            levels = np.linspace(-max_value, max_value, 101)
        cs1 = ax.contourf(X, Y, field_up, levels, cmap=cmap)
        # Lower part
        ax.contourf(X, - Y, field_down, levels, cmap=cmap)
        if contour:
            if cmap == 'Blues':
                clevels = np.array([0.2, 0.5, 0.8]) * max_value
            else:
                clevels = np.array([- 0.8, - 0.2, 0.2, 0.8]) * max_value
            ax.contour(X, Y, field_up, levels=clevels, colors='k', linewidths=0.9)
            # Lower part
            ax.contour(X, -Y, field_down, levels=clevels, colors='k', linewidths=0.9)

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

        fig.colorbar(cs1, ax=ax, pad=0.05, fraction=fraction_cbar, aspect=aspect,
            ticks=field_ticks)

    if geom == 'xr':
        ax.set_yticks([-ymax, -ymax / 2, 0, ymax / 2, ymax])

    # Apply same formatting to x and y axis with scientific notation
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.set_aspect("equal")
    ax.set_title(title)

    ax.plot(X[0, :], np.zeros_like(X[0, :]), 'k', lw=0.7)

    # Remove axis ticks
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def main():
    parser = argparse.ArgumentParser(description='Streamer post-processing globals')
    parser.add_argument('-c', '--config', type=str,
                        help='Config filename', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    fig_dir = Path('figures') / cfg['fig_dir']
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load the numpy arrays
    solut_paths = cfg['solut_paths']
    solut_labels = cfg['solut_labels']

    # Limitations of colorbars
    ne_ticks = cfg['ne_ticks']
    Emax = cfg['Emax']

    # Creation of grid
    mesh = Mesh(cfg)

    # Instants wanted
    instants = cfg['instants']
    ninstants = len(instants)

    for instant in instants:
        fig, axes = plt.subplots(nrows=2, figsize=(6, 6))
        axes = axes.reshape(-1)
        # Load solutions
        normE_up = np.load(Path(solut_paths[0]) / f'normE_{instant:04d}.npy')
        normE_down = np.load(Path(solut_paths[1]) / f'normE_{instant:04d}.npy')
        nd_up = np.load(Path(solut_paths[0]) / f'nd_{instant:04d}.npy')
        nd_down = np.load(Path(solut_paths[1]) / f'nd_{instant:04d}.npy')

        # Plotting
        plot_ax_scalar(fig, axes[0], mesh.X, mesh.Y, nd_up[0], nd_down[0], r"$n_e$", cmap_scale='log',
                            field_ticks=ne_ticks)
        plot_ax_scalar(fig, axes[1], mesh.X, mesh.Y, normE_up, normE_down, r"$|\mathbf{E}|$",
                            cmap='Blues', max_value=Emax)

        plt.tight_layout()
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.savefig(fig_dir / f'comp_2D_{instant:04d}', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # Combined plot
    combined = cfg['combined'] == 'yes'
    if combined:
        fig, axes = plt.subplots(nrows=2, ncols=ninstants, figsize=(4 * ninstants + 1, 5))
        for i_inst, instant in enumerate(instants):
            # Load solutions
            normE_up = np.load(Path(solut_paths[0]) / f'normE_{instant:04d}.npy')
            normE_down = np.load(Path(solut_paths[1]) / f'normE_{instant:04d}.npy')
            nd_up = np.load(Path(solut_paths[0]) / f'nd_{instant:04d}.npy')
            nd_down = np.load(Path(solut_paths[1]) / f'nd_{instant:04d}.npy')

            # Plotting
            plot_ax_scalar(fig, axes[0][i_inst], mesh.X, mesh.Y, nd_up[0], nd_down[0], r"$n_e$", cmap_scale='log',
                                field_ticks=ne_ticks, cbar=False)
            plot_ax_scalar(fig, axes[1][i_inst], mesh.X, mesh.Y, normE_up, normE_down, r"$|\mathbf{E}|$",
                                cmap='Blues', max_value=Emax, cbar=False)

        plt.tight_layout()
        plt.savefig(fig_dir / f'comp_2D', dpi=200, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()