# Post-processing metrics.h5 files from network trainings
# through use of a yaml config file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import yaml
from pathlib import Path

label_dict = {'Eresidual': r'$||\mathbf{E}_\mathrm{out} - \mathbf{E}_\mathrm{target}||_1$',
        'Einf_norm': r'$||\mathbf{E}_\mathrm{out} - \mathbf{E}_\mathrm{target}||_\infty$',
        'residual': r'$||\phi_\mathrm{out} - \phi_\mathrm{target}||_1$',
        'inf_norm': r'$||\phi_\mathrm{out} - \phi_\mathrm{target}||_1$',
        'loss': '$\mathcal{L}$', 'DirichletBoundaryLoss': '$\mathcal{L}_D$',
        'LaplacianLoss':'$\mathcal{L}_L$'}

def ax_prop(ax, ylabel):
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()

def main():
    # Parse cli argument
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', type=str, required=True)
    args = args.parse_args()

    # convert yml file to dict
    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    # Retrieve data from h5
    data_dir = Path(cfg['data_dir'])
    fig_dir = data_dir / 'figures' / cfg['fig_dir']
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_networks = dict()
    for nn_name in cfg['networks']:
        data_networks[nn_name] = pd.read_hdf(data_dir / nn_name / 'metrics.h5', key=cfg['type'])
    
    # Plotting
    for figname in cfg['plot']:
        naxes = len(cfg['plot'][figname])
        fig, axes = plt.subplots(ncols=naxes, figsize=(5 * naxes, 5))
        if naxes == 1:
            axes = [axes]
        for i_qty, qty in enumerate(cfg['plot'][figname]):
            for nn_name, nn_data in data_networks.items():
                axes[i_qty].plot(nn_data.index, nn_data[qty], label=cfg['networks'][nn_name])
                ax_prop(axes[i_qty], label_dict[qty])
        
        fig.savefig(fig_dir / figname, bbox_inches='tight')

if __name__ == '__main__':
    main()