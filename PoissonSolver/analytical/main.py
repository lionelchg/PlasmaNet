########################################################################################################################
#                                                                                                                      #
#                                        2D Poisson analytical solution tests                                          #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import yaml
import scipy.constants as co

from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
from PlasmaNet.poissonsolver.analytical import PoissonAnalytical
from PlasmaNet.common.profiles import gaussian
from PlasmaNet.common.utils import create_dir


if __name__ == '__main__':
    fig_dir = 'figures/rhs_class/'
    create_dir(fig_dir)
        
    with open('poisson_ls_xy.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    poisson = PoissonLinSystem(cfg)

    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    ones_x, ones_y = np.ones(poisson.nnx), np.ones(poisson.nny)

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2

    # interior rhs
    physical_rhs = gaussian(poisson.X, poisson.Y, ni0, x0, y0, 
                            sigma_x, sigma_y) * co.e / co.epsilon_0

    poisson.solve(physical_rhs.reshape(-1), zeros_x, zeros_x, zeros_y, zeros_y)
    poisson.plot_2D(fig_dir + '2D')
    poisson.plot_1D2D(fig_dir + 'full')

    # Declaration of class Poisson Analytical
    poisson_th = PoissonAnalytical(cfg)

    # Solve rhs problem
    poisson_th.compute_sol(physical_rhs, zeros_y, zeros_y, zeros_x, zeros_x)
    poisson_th.plot_2D(fig_dir + '2D_th_1')
    poisson_th.plot_1D2D(fig_dir + 'full_th_1')

    # Solve down, up, left, right constant bc problem
    poisson_th.compute_sol(ones_y, zeros_y, zeros_x, zeros_x)
    poisson_th.plot_2D(fig_dir + '2D_th_2')
    poisson_th.plot_1D2D(fig_dir + 'full_th_2')

    poisson_th.compute_sol(zeros_y, ones_y, zeros_x, zeros_x)
    poisson_th.plot_2D(fig_dir + '2D_th_3')
    poisson_th.plot_1D2D(fig_dir + 'full_th_3')

    poisson_th.compute_sol(zeros_y, zeros_y, ones_x, zeros_x)
    poisson_th.plot_2D(fig_dir + '2D_th_4')
    poisson_th.plot_1D2D(fig_dir + 'full_th_4')

    poisson_th.compute_sol(zeros_y, zeros_y, zeros_x, ones_x)
    poisson_th.plot_2D(fig_dir + '2D_th_5')
    poisson_th.plot_1D2D(fig_dir + 'full_th_5')
