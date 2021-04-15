import argparse
import yaml
import os
import numpy as np
import scipy.constants as co

# From PlasmaNet
from PlasmaNet.poissonsolver.network import PoissonNetwork
from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
import PlasmaNet.common.profiles as pf

def plot_every(plt_evr, it):
    """ Useful function to plot every n its
    """
    if it%plt_evr==0:
        plot = True
    else:
        plot = False
    return plot

def field_diff(pot_nn, pot_data, rhs_data, poisson):
    """ Computes the E_norm and Lapl of potential

    Args:
        pot_nn (array): Potential predicted by the nn
        pot_data (array): Loaded potential from Linsolver
        rhs_data (array): Loaded rhs from Linsolver
        poisson (class): Poisson object

    Returns:
        tuple: abs difference of pot, E and lapl (and normalized by the max value!)
    """

    E_field_nn = poisson.E_field
    E_norm_nn = np.sqrt(E_field_nn[0]**2 + E_field_nn[1]**2)
    lapl_nn = poisson.lapl

    poisson.potential = pot_data
    E_field = poisson.E_field
    E_norm = np.sqrt(E_field[0]**2 + E_field[1]**2)

    return np.abs(pot_nn - pot_data)/(np.max(np.abs(pot_data))), \
           np.abs(E_norm_nn - E_norm)/(np.max(np.abs(E_norm))), \
           np.abs(lapl_nn - rhs_data)/(np.max(np.abs(rhs_data)))

def metric_calc(diff, test_len):
    """ 3 studied metrics, l1, l2 and linf
    """

    l1 = np.sum(np.abs(diff))/ test_len
    l2 = np.sum(diff**2)/ test_len
    linf = np.sum(np.max(diff, axis=0))/ test_len

    return l1, l2, linf

def run_cases(case_dir, poisson, dataset, plt_evr, *args):
    """ 
    Args:
        case_dir (str): directory to save plots
        poisson (class): object for poisson
        dataset (str): dataset path
        plt_evr (int): plotting variable

    """

    # Load Testing dataset
    dataset_rhs = np.load(os.path.join(dataset, 'physical_rhs.npy'))
    dataset_pot = np.load(os.path.join(dataset, 'potential.npy'))

    e_diff = np.zeros_like(dataset_pot)
    lapl_diff = np.zeros_like(dataset_pot)
    pot_diff = np.zeros_like(dataset_pot)

    l1 = np.zeros(3)
    l2 = np.zeros(3)
    linf = np.zeros(3)

    test_len = len(dataset_rhs[:,0,0])

    for i in range(test_len):
        poisson.run_case(case_dir, dataset_rhs[i], plot_every(plt_evr, i), i)
        pot_diff[i], e_diff[i], lapl_diff[i] = field_diff(poisson.potential, dataset_pot[i], dataset_rhs[i], poisson)

    diffs = [pot_diff, e_diff, lapl_diff]

    for i, diff in enumerate(diffs):
        l1[i], l2[i], linf[i] = metric_calc(diff, test_len)

    return l1, l2, linf

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PoissonNetwork runs')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args.add_argument('-d', '--datadir', default=None, type=str,
                      help='Dataset directory (should be .npy)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    with open('studied_networks.yml', 'r') as yaml_stream:
        config_net = yaml.safe_load(yaml_stream)

    # Testing and plotting options
    plt_evr = config['plot_every']
    dataset_loc = config['dataset'] 
    basecase_dir = config['network']['casename']
    basecase_resume = config['network']['resume'] 
    config['network']['eval'] = config['linsystem']

    # Evalutions
    networks = config['evaluation']['networks']
    datasets = config['evaluation']['datasets']
    # l1 values
    l1_values = np.zeros((3, len(datasets), len(networks)))
    l2_values = np.zeros((3, len(datasets), len(networks)))
    linf_values = np.zeros((3, len(datasets), len(networks)))
    metrics = np.zeros((3, 3*len(datasets), len(networks)))

    network_names = ['UNet3_u', 'UNet4_u', 'UNet5_u',
                     'UNet3_d', 'UNet4_d', 'UNet5_d',
                     'UNet3_wide_u', 'UNet4_wide_u', 'UNet5_wide_u',
                     'UNet3_wide_d', 'UNet4_wide_d', 'UNet5_wide_d']

    for i, network in enumerate(networks):
        for j, t_dataset in enumerate(datasets):

            config['network']['arch'] = config_net[network_names[network-1]]
            config['network']['arch']['input_res'] = config['network']['globals']['nnx']
            config['network']['resume'] = os.path.join(basecase_resume, 'config_{}/{}'.format(network, t_dataset))

            poisson_nn = PoissonNetwork(config['network'])
            case_dir = os.path.join(basecase_dir, 'config_{}/{}/'.format(network, t_dataset))
            l1_values[:, j, i], l2_values[:, j, i], linf_values[:, j, i] = run_cases(case_dir, poisson_nn, dataset_loc, plt_evr)

            print('Pot Error levels: l1 = {:.03f}, l2 = {:.03f}, l_inf = {:.03f}'.format(l1_values[0, j, i], l2_values[0, j, i], linf_values[0, j, i]))
            print('Enorm Error levels: l1 = {:.03f}, l2 = {:.03f}, l_inf = {:.03f}'.format(l1_values[1, j, i], l2_values[1, j, i], linf_values[1, j, i]))
            print('Lapla Error levels: l1 = {:.03f}, l2 = {:.03f}, l_inf = {:.03f}'.format(l1_values[2, j, i], l2_values[2, j, i], linf_values[2, j, i]))

    metrics[:, 0:3] = l1_values
    metrics[:, 3:6] = l2_values
    metrics[:, 6:9] = linf_values 

    np.save(basecase_dir + 'l_1.npy', l1_values)
    np.save(basecase_dir + 'l_2.npy', l2_values)
    np.save(basecase_dir + 'l_inf.npy', linf_values)
    np.save(basecase_dir + 'metrics.npy', metrics)

    # DataFrame using arrays.
    import pandas as pd

    iterables = [ ["slim", "wide"], ["upsample", "deconvolution"], ["3_scales", "4_scales", "5_scales"]]
    iterables_2 = [["l_1", "l_2", "l_inf"], ["random_4", "random_8", "random_16"]]
    index = pd.MultiIndex.from_product(iterables, names=["scale", "width", "up_type"])
    index2 = pd.MultiIndex.from_product(iterables_2, names=["dataset", "metric"])

    df_pot = pd.DataFrame(metrics[0], index=index2, columns=index)
    df_Enorm = pd.DataFrame(metrics[1], index=index2, columns=index)
    df_lapl = pd.DataFrame(metrics[2], index=index2, columns=index)




  