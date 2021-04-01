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
with open(os.path.join(base_dir, 'poisson_ls_xy.yml')) as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
cfg['bcs'] = 'neumann'

poisson = PoissonLinSystem(cfg)

rtol = 1e-10
atol = 1e-10

bcs = {}

class TestRhs:
    def test_dipole(self):
        case_dir = os.path.join(base_dir, 'cases/neumann/dipole/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy')
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, bcs)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
    
    def test_quadrupole(self):
        case_dir = os.path.join(base_dir, 'cases/neumann/quadrupole/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy')
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, bcs)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
    
