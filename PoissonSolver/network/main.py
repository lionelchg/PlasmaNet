import argparse
import yaml
import numpy as np
import scipy.constants as co

# From PlasmaNet
from PlasmaNet.poissonsolver.network import PoissonNetwork
from PlasmaNet.poissonsolver.poisson import PoissonLinSystem, run_case
import PlasmaNet.common.profiles as pf
from PlasmaNet.common.utils import create_dir

def run_case_nn(poisson: PoissonNetwork, case_dir: str, physical_rhs: np.ndarray,
             Lx: float, plot: bool):
    """ Run a Poisson linear system case

    :param poisson: PoissonNetwork object
    :type poisson: PoissonNetwork
    :param case_dir: Case directory
    :type case_dir: str
    :param physical_rhs: physical rhs
    :type physical_rhs: np.ndarray
    :param Lx: length of the domain
    :type Lx: float
    :param plot: logical for plotting
    :type plot: bool
    """
    create_dir(case_dir)
    poisson.solve(physical_rhs, Lx)
    poisson.save(case_dir)
    if plot:
        fig_dir = case_dir + 'figures/'
        create_dir(fig_dir)
        poisson.plot_2D(fig_dir + '2D')
        poisson.plot_1D2D(fig_dir + 'full')


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

    # Create neural network instance and linear system instance
    poisson_nn = PoissonNetwork(config['network'])
    poisson_ls = PoissonLinSystem(config['linsystem'])

    # Case directory
    basecase_dir = 'cases/'

    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2
    physical_rhs = pf.gaussian(poisson_ls.X, poisson_ls.Y, 
                ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0

    # Solve using linear system
    ls_case_dir = f'{basecase_dir}linsystem/gaussian/'
    zeros_x, zeros_y = np.zeros(poisson_ls.nnx), np.zeros(poisson_ls.nny)
    pot_bcs = {'left':zeros_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}
    run_case(poisson_ls, ls_case_dir, physical_rhs, pot_bcs, plot)

    # Solve using neural network
    nn_case_dir = f'{basecase_dir}network/gaussian/'
    run_case_nn(poisson_nn, nn_case_dir, physical_rhs, poisson_nn.Lx, plot)

