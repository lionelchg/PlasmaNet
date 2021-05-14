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

# From PlasmaNet
from PlasmaNet.poissonsolver.network import PoissonNetwork

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PoissonNetwork runs')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Neural network configuration
    config['network']['eval'] = config['eval']
    poisson_nn = PoissonNetwork(config['network'])

    # datasets studied
    datasets = config['datasets']
    ndatasets = len(datasets)
    metrics_ds = dict()

    # datadir
    data_dir = Path(config['network']['casename']) / 'datasets'

    # Evaluate on the datasets specified in config
    for ds_name, ds_loc in datasets.items():
        metrics_tmp = poisson_nn.evaluate(ds_loc, data_dir / ds_name, plot=False)
        metrics_ds[ds_name] = deepcopy(metrics_tmp)
    
    # Reformat to one dataframe per metrics with figure plotting
    metrics = dict()
    nmetrics = len(config['network']['metrics'])
    fig, axes = plt.subplots(ncols=nmetrics, figsize=(4 * nmetrics, 4))
    for imetric, metric_name in enumerate(config['network']['metrics']):
        tmp_df = pd.DataFrame(columns=datasets.keys())
        tmp_dict = dict()
        for ds_name in datasets.keys():
            tmp_dict[ds_name] = metrics_ds[ds_name]._data.average[metric_name]
        tmp_df.loc['UNet5'] = tmp_dict

        # Save figures for each metric
        tmp_df.T.plot.bar(ax=axes[imetric], rot=0)
        axes[imetric].set_title(metric_name)

        # Pass the metrics to a global dict
        metrics[metric_name] = tmp_df
    
    fig.tight_layout()
    fig.savefig(data_dir / 'metrics', bbox_inches='tight')

    