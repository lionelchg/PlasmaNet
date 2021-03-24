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

from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
from PlasmaNet.common.profiles import gaussian
from PlasmaNet.common.utils import create_dir


def th_solution(X, Y, n, m, Lx, Ly):
    return np.sin(n * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)

if __name__ == '__main__':
    fig_dir = 'figures/convergence/'
    create_dir(fig_dir)

    with open('poisson_ls_xy.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    cfg['bcs'] = 'dirichlet'
    
    # creating the rhs
    ni0 = 1e+11
    n, m = 5, 5
    
    nnxs = np.array([51, 101, 201, 401])
    errors = np.zeros(len(nnxs))

    for i_err, nnx in enumerate(nnxs):
        print(f'nnx = {nnx:d}')
        nny = nnx
        zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)
        pot_bcs = {'left':zeros_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}

        # Change the configuration dict
        cfg['nnx'] = nnx
        cfg['nny'] = nny

        poisson = PoissonLinSystem(cfg)

        # interior rhs
        physical_rhs = ni0 * th_solution(poisson.X.reshape(-1), poisson.Y.reshape(-1),
                                         n, m, poisson.Lx, poisson.Ly)
        potential_th = (physical_rhs.reshape(nny, nnx) / ((n * np.pi / poisson.Lx)**2 + (m * np.pi / poisson.Ly)**2))
        poisson.solve(physical_rhs, pot_bcs)
        poisson.plot_1D2D(fig_dir + f'sol_{nnx}')
        errors[i_err] = poisson.L2error(potential_th)

        poisson.potential = potential_th
        poisson.plot_1D2D(fig_dir + f'th_{nnx}')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(nnxs, errors)
    ax.grid(True)
    ax.set_xlabel(r'$n_\mathrm{pts}$')
    ax.set_ylabel(r'$\varepsilon_2$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(fig_dir + 'L2_error', bbox_inches='tight')
    plt.close(fig)
