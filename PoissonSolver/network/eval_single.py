############################################################################################################
#                                                                                                          #
#                                Network evaluation on fixed profiles                                      #
#                                                                                                          #
#                                  Lionel Cheng, CERFACS, 16.04.2021                                       #
#                                                                                                          #
############################################################################################################

import os
import argparse
import yaml
from pathlib import Path
import numpy as np
import scipy.constants as co

# From PlasmaNet
from PlasmaNet.poissonsolver.network import PoissonNetwork
import PlasmaNet.common.profiles as pf

def run_cases(basecase_dir, poisson, plot):
    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2
    physical_rhs = pf.gaussian(poisson.X, poisson.Y, 
                ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0

    # Gaussian
    case_dir = basecase_dir / 'gaussian'
    poisson.run_case(case_dir, physical_rhs, plot)

    # Step
    case_dir = basecase_dir / 'step'
    physical_rhs = pf.step(poisson.X, poisson.Y, 
                    ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, plot)

    # Two Gaussians
    case_dir = basecase_dir / 'two_gaussians'
    x01, y01 = 0.4e-2, 0.5e-2
    x0, y0 = 0.6e-2, 0.5e-2    
    physical_rhs = pf.two_gaussians(poisson.X, poisson.Y, 
                    ni0, x0, y0, sigma_x, sigma_y, x01, y01, sigma_x, sigma_y) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, plot)

    # Random 4
    case_dir = basecase_dir / 'random_4'
    physical_rhs = pf.random2D(poisson.X, poisson.Y, ni0, 4) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, plot)
    fig_dir = case_dir / 'figures'
    poisson.plot_difference_modes(fig_dir / '2D_modes')

    # Random 8
    case_dir = basecase_dir / 'random_8'
    physical_rhs = pf.random2D(poisson.X, poisson.Y, ni0, 8) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, plot)
    fig_dir = case_dir / 'figures'
    poisson.plot_difference_modes(fig_dir / '2D_modes')

    # Random 16
    case_dir = basecase_dir / 'random_16'
    physical_rhs = pf.random2D(poisson.X, poisson.Y, ni0, 16) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, plot)
    fig_dir = case_dir / 'figures'
    poisson.plot_difference_modes(fig_dir / '2D_modes')

    # sin_2D
    case_dir = basecase_dir / 'sin_2D'
    physical_rhs = pf.sin2D(poisson.X, poisson.Y, ni0, poisson.Lx, poisson.Ly, 4, 4) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, plot)

    # Dipole
    case_dir = basecase_dir / 'dipole'
    x0, y0 = 0.6e-2, 0.5e-2
    x01, y01 = 0.4e-2, 0.5e-2
    physical_rhs = pf.gaussians(poisson.X, poisson.Y, 
                    [ni0, x0, y0, sigma_x, sigma_y, -ni0, x01, y01, sigma_x, sigma_y]) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, plot)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PoissonNetwork runs')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)
    
    if 'networks' in config:
        base_path = config['network']['casename']
        for net in config['networks']:
            config['network']['eval'] = config['eval']
            config['network']['resume'] = config['networks'][net]['resume']
            config['network']['arch'] = config['networks'][net]['arch']
            config['network']['casename'] = os.path.join(base_path, net, 'single')

            poisson_nn = PoissonNetwork(config['network'])
            np.random.seed(101)
            run_cases(Path(config['network']['casename']), poisson_nn, plot=True)
    else:
        config['network']['eval'] = config['eval']
        poisson_nn = PoissonNetwork(config['network'])
        run_cases(Path(config['network']['casename']) / 'single', poisson_nn, plot=True)
