########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson network convergence                                            #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

from logging import error
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
from PlasmaNet.poissonsolver.poisson import DatasetPoisson
from PlasmaNet.poissonsolver.network import PoissonNetwork
from PlasmaNet.common.profiles import gaussian
from PlasmaNet.common.operators_numpy import grad
from PlasmaNet.common.utils import create_dir

sns.set_context('notebook', font_scale=1.0)


def ax_prop(ax, title):
    ax.grid(True)
    ax.set_xlabel(r'$n_\mathrm{pts}$')
    ax.set_ylabel(r'$\varepsilon$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()

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

    # creating the rhs amplitude and the studied modes
    ni0 = 1e+11
    rhs0 = ni0 * co.e / co.epsilon_0
    modes = ((1, 1), (1, 2), (2, 1), (2, 2), (4, 4), (6, 6), (10, 10))
    
    # Different studied resolutions
    nnxs = np.array([51, 101, 201, 401, 701, 1001, 2001])
    errors = {'L1': {}, 'L2': {}, 'Linf': {}}
    for key in errors:
        for (mode_n, mode_m) in modes:
            errors[key][f'({mode_n:d}, {mode_m:d})'] = {'potential': np.zeros(len(nnxs)),
                'E_field': np.zeros(len(nnxs))}
    
    # Loop on different resolutions
    for i_nnx, nnx in enumerate(nnxs):
        print(f'Case resolution nnx = {nnx:d}')
        nny = nnx

        # Change the configuration dict and initialize new poisson object with good res
        cfg['network']['eval']['nnx'] = nnx
        cfg['network']['eval']['nny'] = nny
        cfg['network']['arch']['args']['input_res'] = nnx
        poisson = PoissonNetwork(cfg['network'])
        poisson.case_config(cfg['network']['eval'])

        # interior rhs for exact solution it is the mode specified by nmax and mmax above
        for (mode_n, mode_m) in modes:
            physical_rhs = (ni0 * dirichlet_mode(poisson.X, poisson.Lx, mode_n) * 
                    dirichlet_mode(poisson.Y, poisson.Ly, mode_m)) * co.e / co.epsilon_0
            potential_th = physical_rhs / ((mode_n * np.pi / poisson.Lx)**2 + (mode_m * np.pi / poisson.Ly)**2)
            E_field_th = - grad(potential_th, poisson.dx, poisson.dy, nnx, nny)
            poisson.solve(physical_rhs)
            if i_nnx == len(nnxs) - 1:
                poisson.plot_1D2D(fig_dir / f'sol_{nnx}_{mode_n}_{mode_m}')
            for key in errors:
                errors[key][f'({mode_n:d}, {mode_m:d})']['potential'][i_nnx] = getattr(poisson, f'{key}error_pot')(potential_th)
                errors[key][f'({mode_n:d}, {mode_m:d})']['E_field'][i_nnx] = getattr(poisson, f'{key}error_E')(E_field_th)

        poisson.potential = potential_th
        # poisson.plot_1D2D(fig_dir / f'th_{nnx}')
    
    # Creation of L1, L2 and Linf error figs
    for error_kind in errors:
        fig, axes = plt.subplots(ncols=2, figsize=(8, 5))
        for (mode_n, mode_m) in modes:
            axes[0].plot(nnxs, errors[error_kind][f'({mode_n:d}, {mode_m:d})']['potential'], label=f'({mode_n:d}, {mode_m:d})')
        ax_prop(axes[0], 'Potential')
        for (mode_n, mode_m) in modes:
            axes[1].plot(nnxs, errors[error_kind][f'({mode_n:d}, {mode_m:d})']['E_field'], label=f'({mode_n:d}, {mode_m:d})')
        ax_prop(axes[1], 'Electric field')
        fig.tight_layout()
        fig.savefig(fig_dir / f'{error_kind}_error', bbox_inches='tight')
        plt.close(fig)
