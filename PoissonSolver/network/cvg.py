########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson network convergence                                            #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 15.09.2021                                           #
#                                                                                                                      #
########################################################################################################################

from logging import error
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants as co
import seaborn as sns
from pathlib import Path

# From PlasmaNet
from PlasmaNet.poissonsolver.analytical import dirichlet_mode
from PlasmaNet.poissonsolver.network import PoissonNetwork
from PlasmaNet.common.operators_numpy import grad

def ax_prop(ax, ylabel):
    ax.grid(True)
    ax.set_xlabel(r'$n_\mathrm{pts}$')
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')

if __name__ == '__main__':
    # Plotting options for aesthetics
    sns.set_context('notebook', font_scale=1.1)
    lines_params = {'basic': {'linewidth': 2, 'markersize': 6}}
    line_style = {'marker':'o'}
    mpl.rc('lines', **lines_params['basic'])
    # mpl.rc('savefig', format='pdf')

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
    modes = ((1, 1), (1, 2), (2, 1), (2, 2), (6, 6), (10, 10))
    modes_style = [{'color': 'darkblue', 'marker': 'o'}, {'color': 'royalblue', 'marker':'o', 'linestyle':'dashed'}, 
    {'color': 'royalblue', 'marker': '^', 'linestyle':'dashed'}, {'color': 'royalblue', 'marker': 's', 'linestyle':'dashed'}, 
    {'color': 'firebrick', 'marker': 'o', 'linestyle':'dotted'}, {'color': 'firebrick', 'marker': '^', 'linestyle':'dotted'}]
    
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
            # if i_nnx == len(nnxs) - 1:
            #     poisson.plot_1D2D(fig_dir / f'sol_{nnx}_{mode_n}_{mode_m}')
            for key in errors:
                errors[key][f'({mode_n:d}, {mode_m:d})']['potential'][i_nnx] = getattr(poisson, f'{key}error_pot')(potential_th)
                errors[key][f'({mode_n:d}, {mode_m:d})']['E_field'][i_nnx] = getattr(poisson, f'{key}error_E')(E_field_th)

            poisson.potential = potential_th
            
            # Errors in percentage
            print(f'\n nnx = {nnx:d} - (n, m) = ({mode_n:d}, {mode_m:d})')
            print(f"L1_pot = {errors['L1'][f'({mode_n:d}, {mode_m:d})']['potential'][i_nnx] / poisson.L1_pot():.2e}")
            print(f"L2_pot = {errors['L2'][f'({mode_n:d}, {mode_m:d})']['potential'][i_nnx] / poisson.L2_pot():.2e}")
            print(f"Linf_pot = {errors['Linf'][f'({mode_n:d}, {mode_m:d})']['potential'][i_nnx] / poisson.Linf_pot():.2e}")
            print(f"L1_E = {errors['L1'][f'({mode_n:d}, {mode_m:d})']['E_field'][i_nnx] / poisson.L1_E():.2e}")
            print(f"L2_E = {errors['L2'][f'({mode_n:d}, {mode_m:d})']['E_field'][i_nnx] / poisson.L2_E():.2e}")
            print(f"Linf_E = {errors['Linf'][f'({mode_n:d}, {mode_m:d})']['E_field'][i_nnx] / poisson.Linf_E():.2e}")
            
        # poisson.plot_1D2D(fig_dir / f'th_{nnx}')
    
    # Creation of L1, L2 and Linf error figs
    for error_kind in errors:
        fig, axes = plt.subplots(ncols=2, figsize=(8, 5))
        for imode, (mode_n, mode_m) in enumerate(modes):
            axes[0].plot(nnxs, errors[error_kind][f'({mode_n:d}, {mode_m:d})']['potential'], label=f'({mode_n:d}, {mode_m:d})', **modes_style[imode])
        ax_prop(axes[0], r'$\epsilon_\phi$')
        for imode, (mode_n, mode_m) in enumerate(modes):
            axes[1].plot(nnxs, errors[error_kind][f'({mode_n:d}, {mode_m:d})']['E_field'], label=f'({mode_n:d}, {mode_m:d})', **modes_style[imode])
        ax_prop(axes[1], r'$\epsilon_\mathbf{E}$')

        # Put global legend for both figures
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', borderaxespad=0.1)
        fig.subplots_adjust(right=0.82)

        # Move left figure
        tmp_pos = fig.axes[0].get_position()
        tmp_pos.x0 -= 0.05
        tmp_pos.x1 -= 0.05
        fig.axes[0].set_position(tmp_pos)

        # fig.tight_layout()
        fig.savefig(fig_dir / f'{error_kind}_error.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(fig_dir / f'{error_kind}_error.png', bbox_inches='tight')
        plt.close(fig)
