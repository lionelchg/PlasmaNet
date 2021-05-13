########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson network convergence                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
import seaborn as sns
from pathlib import Path

from PlasmaNet.poissonsolver.analytical import dirichlet_mode
from PlasmaNet.poissonsolver.network import PoissonNetwork
from PlasmaNet.common.profiles import gaussian
from PlasmaNet.common.operators_numpy import grad
from PlasmaNet.common.utils import create_dir

sns.set_context('notebook', font_scale=1.0)

def ax_prop(ax, title):
    ax.grid(True)
    ax.set_xlabel(r'$n_\mathrm{pts}$')
    ax.set_ylabel(r'$\varepsilon_2$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PoissonNetwork runs')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    cfg['network']['eval'] = cfg['eval']

    fig_dir = Path(cfg['network']['casename']) / 'cvg'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # creating the rhs
    ni0 = 1e+11
    n, m = 4, 4
    
    # Different studied resolutions
    nnxs = np.array([51, 101, 201, 401, 801])
    errors = {'potential': np.zeros(len(nnxs)),
                'E_field': np.zeros(len(nnxs))}
    
    # Loop on different resolutions
    for i_err, nnx in enumerate(nnxs):
        print(f'Case resolution nnx = {nnx:d}')
        nny = nnx

        # Change the configuration dict and initialize new poisson object with good res
        cfg['network']['eval']['nnx'] = nnx
        cfg['network']['eval']['nny'] = nny
        cfg['network']['arch']['args']['input_res'] = nnx
        poisson = PoissonNetwork(cfg['network'])
        poisson.case_config(cfg['network']['eval'])

        # interior rhs for exact solution it is the mode specified by n and m above
        physical_rhs = (ni0 * dirichlet_mode(poisson.X, poisson.Lx, n) * 
                dirichlet_mode(poisson.Y, poisson.Ly, m)) * co.e / co.epsilon_0
        potential_th = physical_rhs / ((n * np.pi / poisson.Lx)**2 + (m * np.pi / poisson.Ly)**2)
        E_field_th = - grad(potential_th, poisson.dx, poisson.dy, nnx, nny)
        poisson.solve(physical_rhs)
        poisson.plot_1D2D(fig_dir / f'sol_{nnx}')

        errors['potential'][i_err] = poisson.L2error_pot(potential_th)
        errors['E_field'][i_err] = poisson.L2error_E(E_field_th)

        poisson.potential = potential_th
        poisson.plot_1D2D(fig_dir / f'th_{nnx}')
    
    # Creation of L2 error fig
    fig, axes = plt.subplots(ncols=2, figsize=(8, 5))
    axes[0].plot(nnxs, errors['potential'])
    ax_prop(axes[0], 'Potential')
    axes[1].plot(nnxs, errors['E_field'])
    ax_prop(axes[1], 'Electric field')
    fig.tight_layout()
    fig.savefig(fig_dir / 'L2_error', bbox_inches='tight')
    plt.close(fig)
