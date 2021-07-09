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
import re

# From PlasmaNet
from PlasmaNet.poissonsolver.network import PoissonNetwork

latex_dict = {
    'Eresidual': r'$||\mathbf{E}_\mathrm{out} - \mathbf{E}_\mathrm{target}||_1$',
    'Einf_norm': r'$||\mathbf{E}_\mathrm{out} - \mathbf{E}_\mathrm{target}||_\infty$',
    'residual': r'$||\phi_\mathrm{out} - \phi_\mathrm{target}||_1$',
    'inf_norm': r'$||\phi_\mathrm{out} - \phi_\mathrm{target}||_1$',
    'phi11': r'$||\phi^{out}_\mathrm{11} - \phi^{target}_\mathrm{11}||_1$',
    'phi12': r'$||\phi^{out}_\mathrm{12} - \phi^{targte}_\mathrm{12}||_1$',
    'phi21': r'$||\phi^{out}_\mathrm{21} - \phi^{target}_\mathrm{21}||_1$',
    'phi22': r'$||\phi^{out}_\mathrm{22} - \phi^{target}_\mathrm{22}||_1$',
}


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
    args.add_argument('-fn', '--filename', default='metrics', type=str,
                      help='Name of the h5 file of the DataFrame in the casename directory')
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

    # Metrics and number of metrics
    metrics = config['network']['metrics']
    nmetrics = len(metrics)

    # Global dataframe
    df_columns = ['nn_name', 'nn_type', 'rf_global', 'nbranches', 'depth', 'ks', 'ds_name', 'ds_type',
                       'test_res', 'train_res', 'metric_name', 'value']
    df = pd.DataFrame(columns=df_columns)

    # Regex for dstype and rf_list
    re_ds_type = re.compile('[^_]+')

    # Filter all the rfs to match closest to rf_list
    rf_list = [50, 75, 100, 150, 200, 300, 400]

    # Evaluate on the datasets specified in config for each network
    for nn_name, nn_cfg in networks_cfg.items():
        config['network']['eval'] = config['eval']
        config['network']['resume'] = nn_cfg['resume']
        config['network']['arch'] = nn_cfg['arch']
        poisson_nn = PoissonNetwork(config['network'])

        # Extract network global properties
        rf_global = poisson_nn.model.rf_global
        nbranches = poisson_nn.model.n_scales
        depths = poisson_nn.model.depths
        ks = poisson_nn.model.kernel_sizes[0]
        test_res = config['eval']['nnx']
        train_res = config['network']['globals']['nnx'] 

        # Loop on datasets
        for ds_name, ds_loc in datasets.items():
            metrics_tmp = poisson_nn.evaluate(ds_loc, data_dir / ds_name, plot=False)
            # Loop on the resulting metrics and create the temporary dict
            for metric_name in metrics:
                tmp_dict = dict()
                tmp_dict['nn_name'] = nn_name
                tmp_dict['nn_type'] = type(poisson_nn.model).__name__
                tmp_dict['rf_global'] = min(rf_list, key=lambda x:abs(x - rf_global))
                tmp_dict['nbranches'] = nbranches
                tmp_dict['depth'] = sum(depths)
                tmp_dict['ks'] = ks
                tmp_dict['ds_name'] = ds_name
                tmp_dict['test_res'] = test_res
                tmp_dict['train_res'] = train_res
                tmp_dict['ds_type'] = re_ds_type.search(ds_name).group()
                tmp_dict['metric_name'] = metric_name
                tmp_dict['value'] = metrics_tmp._data.average[metric_name]
                df = df.append(tmp_dict, ignore_index=True)
    
    # Save the dataframe
    h5filename = args.filename + '.h5'
    df.to_hdf(data_dir / h5filename, key='df', mode='w')

    # Delete combined dataset
    if args.mode == 'combined':
        shutil.rmtree(output_dsname)


if __name__ == '__main__':
    main()
