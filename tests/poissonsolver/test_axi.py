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

from PlasmaNet.poissonsolver.poisson import PoissonLinSystem

base_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(base_dir, 'poisson_ls_xr.yml')) as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

poisson = PoissonLinSystem(cfg)

zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)

rtol = 1e-10
atol = 1e-10

bcs = {'left':zeros_y, 'right':zeros_y, 'top':zeros_x}

class TestRhs:
    def test_gaussian(self):
        case_dir = os.path.join(base_dir, 'cases/axi/gaussian/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy')
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, bcs)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
    
