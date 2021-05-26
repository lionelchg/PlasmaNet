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
        tmp_df.T.plot.bar(ax=axes[imetric], rot=30)
        axes[imetric].set_title(metric_name)

        # Pass the metrics to a global dict
        metrics[metric_name] = tmp_df
    
    fig.tight_layout()
    fig.savefig(data_dir / args.figname, bbox_inches='tight')
    plt.close()

    # Delete combined dataset 
    if args.mode == 'combined':
        shutil.rmtree(output_dsname)

if __name__ == '__main__':
    main()