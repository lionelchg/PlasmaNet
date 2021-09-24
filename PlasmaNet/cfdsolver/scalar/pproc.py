import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path
import seaborn as sns

sns.set_context('notebook', font_scale=1.0)

def ax_prop(ax, xlabel, ylabel):
    ax.grid(True)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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
    nsols = len(solut_paths)

    # colors = ['lightsteelblue', 'royalblue', 'darkblue',
    #             'lightcoral', 'firebrick', 'darkred']
    colors = ['royalblue', 'indianred']
    markers = ['--o', '--^']

    fig, axes = plt.subplots(ncols=2, figsize=(8, 5))
    for isol, solut_path in enumerate(solut_paths):
        gstreamer = np.load(Path(solut_path) / 'globals.npy')
        # gstreamer = gstreamer[:2800, :] # For filtering not filled values yet

        # Scale to ns
        time = gstreamer[:, 0] / 1e-9

        # Put the first iterations to the same value of
        # later in the simulations when the heads of streamer really appeared
        gstreamer[:20, 1:3] =  gstreamer[20, 1:3]

        # Scale to mmm and microjoule
        gstreamer[:, 1:3] = gstreamer[:, 1:3] / 1e-3
        gstreamer[:, 3] = gstreamer[:, 3] / 1e-6

        # Cut positive streamer and negative streamers
        ineg_cut = np.argmax(gstreamer[:, 1] < 0.01)
        ipos_cut = np.argmax(gstreamer[:, 2] > 3.99)

        if ipos_cut == 0: ipos_cut = -1
        if ineg_cut == 0: ineg_cut = -1

        axes[0].plot(time[:ineg_cut], gstreamer[:ineg_cut, 1], markers[isol],
            markevery=int(len(time) / 6), color=colors[isol], lw=2, label=solut_labels[isol])
        axes[0].plot(time[:ipos_cut], gstreamer[:ipos_cut, 2], markers[isol],
            markevery=int(len(time) / 6), color=colors[isol], lw=2)
        axes[1].plot(time, gstreamer[:, 3], markers[isol],
            markevery=int(len(time) / 6), color=colors[isol], lw=2, label=solut_labels[isol])

    ax_prop(axes[0], '$t$ [ns]', '$x$ [mm]')
    ax_prop(axes[1], '$t$ [ns]', r'E [$\mu$J]')

    fig.tight_layout()
    fig.savefig(fig_dir / 'comp', bbox_inches='tight')
    fig.savefig(fig_dir / 'comp.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()