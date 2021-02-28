########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver convergence                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co

from poissonsolver.poisson import Poisson
from poissonsolver.funcs import gaussian


def th_solution(X, Y, n, m, Lx, Ly):
    return np.sin(n * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)

if __name__ == '__main__':
    fig_dir = 'figures/convergence/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    xmin, xmax = 0.0, 0.01
    ymin, ymax = 0.0, 0.01
    Lx, Ly = xmax - xmin, ymax - ymin
    # creating the rhs
    ni0 = 15.7
    n, m = 5, 5
    
    nnxs = np.array([51, 101, 201, 401])
    errors = np.zeros(len(nnxs))

    for i_err, nnx in enumerate(nnxs):
        print(f'nnx = {nnx:d}')
        nny = nnx
        dx, dy = (xmax - xmin) / (nnx - 1), (ymax - ymin) / (nny - 1)
        x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

        zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)

        poisson = Poisson(xmin, xmax, nnx, ymin, ymax, nny, 'cart_dirichlet')
        # interior rhs
        physical_rhs = ni0 * th_solution(poisson.X.reshape(-1), poisson.Y.reshape(-1),
                                         n, m, Lx, Ly)
        potential_th = physical_rhs.reshape(nny, nnx) / ((n * np.pi / Lx)**2 + (m * np.pi / Ly)**2)
        poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)
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
