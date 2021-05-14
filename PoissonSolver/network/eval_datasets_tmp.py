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

    # Evaluate on the datasets specified in config
    for ds_name, ds_loc in datasets.items():
        metrics_tmp = poisson_nn.evaluate(ds_loc, 
            Path(config['network']['casename']) / 'datasets' / ds_name, plot=False)
        metrics_ds[ds_name] = metrics_tmp
    
    # Reformat to one dataframe per metrics
    metrics = dict()
    for metric_name in config['network']['metrics']:
        tmp_df = pd.DataFrame(columns=datasets.keys())
        tmp_dict = dict()
        for ds_name in datasets.keys():
            tmp_dict[ds_name] = metrics_ds[ds_name]._data.average[metric_name]
        tmp_df.loc['UNet5'] = tmp_dict
        metrics[metric_name] = tmp_df
    