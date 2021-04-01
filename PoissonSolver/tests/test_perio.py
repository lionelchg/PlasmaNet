########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import yaml
import numpy as np
import math
import pdb

from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
from PlasmaNet.common.utils import create_dir

def th_solution(X, Y, n, m, Lx, Ly):
    return np.sin(n * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)


base_dir = os.path.dirname(os.path.realpath(__file__))
fig_dir = 'debug_perio/'
create_dir(fig_dir)
with open('poisson_ls_xy.yml') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
cfg['bcs'] = 'perio'

# creating the rhs
ni0 = 1e+11
n, m = 5, 5

nny = nnx = 101
zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)
#pot_bcs = {'left':zeros_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}
pot_bcs = {'perio'}

# Change the configuration dict
cfg['nnx'] = nnx
cfg['nny'] = nny

poisson = PoissonLinSystem(cfg)


rtol = 1e-10
atol = 1e-10


class TestRhs:

    def test_sin(self):
        physical_rhs = ni0 * th_solution(poisson.X, poisson.Y,
                                            n, m, poisson.Lx, poisson.Ly)
        potential_th = (physical_rhs.reshape(nny, nnx) / ((n * np.pi / poisson.Lx)**2 + (m * np.pi / poisson.Ly)**2))
        poisson.solve(physical_rhs, pot_bcs)
        poisson.plot_1D2D(fig_dir + f'sol_{nnx}')
        poisson.potential = potential_th
        poisson.plot_1D2D(fig_dir + f'th_{nnx}')
        assert np.allclose(poisson.potential, potential_th, atol=atol, rtol=rtol)

