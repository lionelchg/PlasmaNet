########################################################################################################################
#                                                                                                                      #
#                                        2D Poisson analytical solution tests                                          #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import yaml
import scipy.constants as co
from pathlib import Path

from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
from PlasmaNet.poissonsolver.analytical import PoissonAnalytical
import PlasmaNet.common.profiles as pf


if __name__ == '__main__':
    fig_dir = Path('figures/precision/')
    fig_dir.mkdir(parents=True, exist_ok=True)
        
    with open('poisson_ls_xy.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    poisson = PoissonLinSystem(cfg)

    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    ones_x, ones_y = np.ones(poisson.nnx), np.ones(poisson.nny)
    pot_bcs = {'left':zeros_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.6e-2, 0.5e-2
    x01, y01 = 0.4e-2, 0.5e-2    

    # interior rhs
    physical_rhs = pf.two_gaussians(poisson.X, poisson.Y, 
                    ni0, x0, y0, sigma_x, sigma_y, x01, y01, sigma_x, sigma_y) * co.e / co.epsilon_0

    # linear system solve and solution norm
    poisson.solve(physical_rhs, pot_bcs)
    norm_pot_ref = np.sqrt(np.sum(poisson.potential**2) / poisson.nnx / poisson.nnx)
    norm_E_ref = np.sqrt(np.sum(poisson.E_field[0]**2 + poisson.E_field[1]) / poisson.nnx / poisson.nnx)
    poisson.plot_1D2D(fig_dir / 'ls')

    # Declaration of class Poisson Analytical
    poisson_series = PoissonAnalytical(cfg)

    # Variation of number of modes
    nmax_modes = [2, 4, 6, 8, 10, 12, 14, 16]
    error_modes = np.zeros((2, len(nmax_modes)))

    for inmax, nmax in enumerate(nmax_modes):
        # Set the new values for analytical solution
        poisson_series.nmax_rhs = nmax
        poisson_series.mmax_rhs = nmax
        
        # Solve rhs problem
        poisson_series.compute_sol(physical_rhs, zeros_y, zeros_y, zeros_x, zeros_x)

        # Compute error in percentage
        error_modes[0, inmax] = 100 * poisson_series.L2error_pot(poisson.potential) / norm_pot_ref
        error_modes[1, inmax] = 100 * poisson_series.L2error_E(poisson.E_field) / norm_E_ref

    # Plot the errors
    fig, ax = plt.subplots()
    ax.plot(nmax_modes, error_modes[0], 'k', label='Potential')
    ax.plot(nmax_modes, error_modes[1], 'b', label='Electric field')
    ax.set_xlabel('$N$')
    ax.set_ylabel('Error (%)')
    ax.grid(True)
    ax.legend()
    ax.set_yscale('log')
    fig.savefig(fig_dir / 'errors')