############################################################################################################
#                                                                                                          #
#                                    Network evaluation on datasets                                        #
#                                                                                                          #
#                                  Lionel Cheng, CERFACS, 16.04.2021                                       #
#                                                                                                          #
############################################################################################################
import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import shutil

# From PlasmaNet
from PlasmaNet.poissonsolver.network import PoissonNetwork

latex_dict = {'Eresidual': r'$||\mathbf{E}_\mathrm{out} - \mathbf{E}_\mathrm{target}||_1$',
            'Einf_norm': r'$||\mathbf{E}_\mathrm{out} - \mathbf{E}_\mathrm{target}||_\infty$',
            'residual': r'$||\phi_\mathrm{out} - \phi_\mathrm{target}||_1$',
            'inf_norm': r'$||\phi_\mathrm{out} - \phi_\mathrm{target}||_1$',
            'phi11': r'$||\phi^{out}_\mathrm{11} - \phi^{target}_\mathrm{11}||_1$',
            'phi12': r'$||\phi^{out}_\mathrm{12} - \phi^{targte}_\mathrm{12}||_1$',
            'phi21': r'$||\phi^{out}_\mathrm{21} - \phi^{target}_\mathrm{21}||_1$',
            'phi22': r'$||\phi^{out}_\mathrm{22} - \phi^{target}_\mathrm{22}||_1$'}

def concatenate_ds(datasets: dict, output_dsname: Path):
    """ Concatenate the datasets specified in dict into a single dataset """
    physical_rhs_list = list()
    potential_list = list()
    for ds_name, ds_loc in datasets.items():
        tmp_rhs = np.load(Path(ds_loc) / 'physical_rhs.npy')
        tmp_pot = np.load(Path(ds_loc) / 'potential.npy')
        physical_rhs_list.append(tmp_rhs)
        potential_list.append(tmp_pot)
    
    physical_rhs_list = np.concatenate(tuple(physical_rhs_list), axis=0)
    potential_list = np.concatenate(tuple(potential_list), axis=0)
    np.save(output_dsname / 'physical_rhs', physical_rhs_list)
    np.save(output_dsname / 'potential', potential_list)

def main():
    args = argparse.ArgumentParser(description='PoissonNetwork runs')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args.add_argument('-m', '--mode', default='separate', type=str,
                      help='Run mode of the dataset (either combined or separated)')
    args.add_argument('-fn', '--figname', default='metrics', type=str,
                      help='Name of the figure in the casename directory')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Neural network configurations
    networks_cfg = config['networks']
    n_networks = len(networks_cfg)

    # datadir
    data_dir = Path(config['network']['casename'])
    data_dir.mkdir(parents=True, exist_ok=True)

    # if combined mode then concatenate datasets
    if args.mode == 'combined':
        output_dsname = data_dir / 'combined'
        output_dsname.mkdir(parents=True, exist_ok=True)
        concatenate_ds(config['datasets'], output_dsname)
        config['datasets'] = {'combined': output_dsname}

    # datasets studied
    datasets = config['datasets']
    ndatasets = len(datasets)
    metrics_ds = dict()

    # Evaluate on the datasets specified in config for each network
    for nn_name, nn_cfg in networks_cfg.items():
        metrics_ds[nn_name] = dict()
        config['network']['eval'] = config['eval']
        config['network']['resume'] = nn_cfg['resume']
        config['network']['arch'] = nn_cfg['arch']
        poisson_nn = PoissonNetwork(config['network'])
        # Loop on datasets
        for ds_name, ds_loc in datasets.items():
            metrics_tmp = poisson_nn.evaluate(ds_loc, data_dir / ds_name, plot=False)
            metrics_ds[nn_name][ds_name] = deepcopy(metrics_tmp)
    
    # Reformat to one dataframe per metrics with figure plotting
    metrics = dict()
    nmetrics = len(config['network']['metrics'])
    fig, axes = plt.subplots(ncols=nmetrics, figsize=(4 * nmetrics, 4))
    if nmetrics == 1:
        axes = [axes]
    for imetric, metric_name in enumerate(config['network']['metrics']):
        tmp_df = pd.DataFrame(columns=datasets.keys())
        for nn_name in networks_cfg.keys():
            tmp_dict = dict()
            for ds_name in datasets.keys():
                tmp_dict[ds_name] = metrics_ds[nn_name][ds_name]._data.average[metric_name]
            tmp_df.loc[nn_name] = tmp_dict

        # Save figures for each metric
        if len(tmp_df.columns) > 1:
            tmp_df.T.plot.bar(ax=axes[imetric], rot=30, legend=False)
        elif len(tmp_df.columns) == 1:
            tmp_df.plot.bar(ax=axes[imetric], rot=30, legend=False)
            axes[imetric].grid(True, alpha=0.3)
        axes[imetric].set_ylabel(latex_dict[metric_name])

        # Set the yticks to scientific format
        scilimx = int(np.log10(max(tmp_df.max())))
        axes[imetric].ticklabel_format(axis='y', style='sci', scilimits=(scilimx, scilimx))

        # Set axis limits manually for coherent plots
        if 'bar_plot_limits' in config['network']:
            limits = config['network']['bar_plot_limits'][imetric]
            axes[imetric].set_ylim(limits[0], limits[1])

        # Create grid
        axes[imetric].grid(True, axis='y', alpha=0.3)

        # Pass the metrics to a global dict
        metrics[metric_name] = tmp_df
    
    # Create one legend for all the subplots when not combined option
    if not len(tmp_df.columns) == 1: 
        handles, labels = axes[0].get_legend_handles_labels()
        if not nmetrics == 1:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.02, 0.8))
        else:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.07, 0.8))

    fig.tight_layout()
    fig.savefig(data_dir / args.figname, bbox_inches='tight')
    plt.close()

    # Delete combined dataset 
    if args.mode == 'combined':
        shutil.rmtree(output_dsname)

if __name__ == '__main__':
    main()