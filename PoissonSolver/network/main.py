import argparse
import yaml
import numpy as np
import scipy.constants as co

# From PlasmaNet
from PlasmaNet.poissonsolver.network import PoissonNetwork
from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
import PlasmaNet.common.profiles as pf

def run_cases(basecase_dir, poisson, *args):
    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2
    physical_rhs = pf.gaussian(poisson.X, poisson.Y, 
                ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0

    # Gaussian
    case_dir = f'{basecase_dir}gaussian/'
    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    pot_bcs = {'left':zeros_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}
    poisson.run_case(case_dir, physical_rhs, *args)

    # Step
    case_dir = f'{basecase_dir}step/'
    physical_rhs = pf.step(poisson.X, poisson.Y, 
                    ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, *args)

    # Two Gaussians
    case_dir = f'{basecase_dir}two_gaussians/'
    x01, y01 = 0.4e-2, 0.5e-2
    x0, y0 = 0.6e-2, 0.5e-2    
    physical_rhs = pf.two_gaussians(poisson.X, poisson.Y, 
                    ni0, x0, y0, sigma_x, sigma_y, x01, y01, sigma_x, sigma_y) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, *args)

    # Random
    case_dir = f'{basecase_dir}random_2D/'
    physical_rhs = pf.random2D(poisson.X, poisson.Y, ni0, 4) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, *args)

    # sin_2D
    case_dir = f'{basecase_dir}sin_2D/'
    physical_rhs = pf.sin2D(poisson.X, poisson.Y, ni0, poisson.Lx, poisson.Ly, 4, 4) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, *args)

    # Dipole
    case_dir = f'{basecase_dir}dipole/'
    x0, y0 = 0.6e-2, 0.5e-2
    x01, y01 = 0.4e-2, 0.5e-2
    physical_rhs = pf.gaussians(poisson.X, poisson.Y, 
                    [ni0, x0, y0, sigma_x, sigma_y, -ni0, x01, y01, sigma_x, sigma_y]) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, *args)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PoissonNetwork runs')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args.add_argument('-d', '--datadir', default=None, type=str,
                      help='Dataset directory (should be .npy)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    plot = True

    # Linear system solver
    poisson_ls = PoissonLinSystem(config['linsystem'])
    zeros_x, zeros_y = np.zeros(poisson_ls.nnx), np.zeros(poisson_ls.nny)
    pot_bcs = {'left':zeros_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}
    basecase_dir = 'cases/linsystem/'
    run_cases(basecase_dir, poisson_ls, pot_bcs, plot)

    # Neural network config_1/random_4
    config['network']['eval'] = config['linsystem']
    poisson_nn = PoissonNetwork(config['network'])
    basecase_dir = 'cases/network/config_1/random_4/'
    run_cases(basecase_dir, poisson_nn, plot)

    # Neural network config_2/random_4
    config['network']['resume'] = config['network']['resume'].replace('config_1', 'config_2')
    poisson_nn = PoissonNetwork(config['network'])
    basecase_dir = 'cases/network/config_2/random_4/'
    run_cases(basecase_dir, poisson_nn, plot)