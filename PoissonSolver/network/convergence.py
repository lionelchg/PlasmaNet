########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver convergence                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
import seaborn as sns

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
    fig_dir = 'figures/cvg/config_2/random_4/'
    create_dir(fig_dir)

    with open('config.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    cfg['network']['eval'] = cfg['linsystem']

    # creating the rhs
    ni0 = 1e+11
    n, m = 4, 4
    
    nnxs = np.array([51, 101, 201, 401, 801])
    errors = {'potential': np.zeros(len(nnxs)),
                'E_field': np.zeros(len(nnxs))}

    poisson = PoissonNetwork(cfg['network'])

    for i_err, nnx in enumerate(nnxs):
        print(f'nnx = {nnx:d}')
        nny = nnx

        # Change the configuration dict
        cfg['network']['eval']['nnx'] = nnx
        cfg['network']['eval']['nny'] = nny
        poisson.case_config(cfg['network']['eval'])

        # interior rhs
        physical_rhs = (ni0 * dirichlet_mode(poisson.X, poisson.Lx, n) * 
                dirichlet_mode(poisson.Y, poisson.Ly, m)) * co.e / co.epsilon_0
        potential_th = physical_rhs / ((n * np.pi / poisson.Lx)**2 + (m * np.pi / poisson.Ly)**2)
        E_field_th = - grad(potential_th, poisson.dx, poisson.dy, nnx, nny)
        poisson.solve(physical_rhs)
        poisson.plot_1D2D(fig_dir + f'sol_{nnx}')

        errors['potential'][i_err] = poisson.L2error_pot(potential_th)
        errors['E_field'][i_err] = poisson.L2error_E(E_field_th)

        poisson.potential = potential_th
        poisson.plot_1D2D(fig_dir + f'th_{nnx}')
    
    fig, axes = plt.subplots(ncols=2, figsize=(8, 5))
    axes[0].plot(nnxs, errors['potential'])
    ax_prop(axes[0], 'Potential')
    axes[1].plot(nnxs, errors['E_field'])
    ax_prop(axes[1], 'Electric field')
    fig.tight_layout()
    fig.savefig(fig_dir + 'L2_error', bbox_inches='tight')
    plt.close(fig)
