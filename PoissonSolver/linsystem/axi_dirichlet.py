########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import yaml

from PlasmaNet.common.profiles import random1D, random2D
from PlasmaNet.common.utils import create_dir
from PlasmaNet.poissonsolver.poisson import PoissonLinSystem

def V_L(x, r, V_0, b, E_0):
    d_s = np.sqrt((b + x)**2 + r**2)
    return V_0 * b / d_s - E_0 * (1 - (b / d_s)**3) * (x + b)

if __name__ == '__main__':
    basecase_dir = 'cases/dirichlet/laplace/'
    plot = True

    with open('poisson_ls_xr.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    poisson = PoissonLinSystem(cfg)

    zero_rhs = np.zeros_like(poisson.X)

    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    ones_x, ones_y = np.ones(poisson.nnx), np.ones(poisson.nny)
    linear_x = np.linspace(0, 1.0, poisson.nnx)
    linear_y = np.linspace(0, 1.0, poisson.nny)

    case_dir = f'{basecase_dir}linear_pot_x/'
    Vmax = 100.0
    pot_bcs = {'left': zeros_y, 'right': Vmax * ones_y, 'top': Vmax * linear_x}
    poisson.run_case(case_dir, zero_rhs, pot_bcs, plot)

    # Electrode parameters
    V_0 = 6.5e3
    b = 1e-3
    E_0 = 1.0e6
    left_elec = V_L(0, poisson.y, V_0, b, E_0)
    right_elec = V_L(poisson.xmax, poisson.y, V_0, b, E_0)
    top_elec = V_L(poisson.x, poisson.ymax, V_0, b, E_0)
    case_dir = f'{basecase_dir}spherical_elec/'
    pot_bcs = {'left': left_elec, 'right': right_elec, 'top': top_elec}
    poisson.run_case(case_dir, zero_rhs, pot_bcs, plot)