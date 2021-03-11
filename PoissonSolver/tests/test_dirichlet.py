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
import scipy.constants as co

from PlasmaNet.common.utils import create_dir
from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
import PlasmaNet.common.profiles as pf

base_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(base_dir, 'poisson_ls_xy.yml')) as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

poisson = PoissonLinSystem(cfg)

zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
ones_x, ones_y = np.ones(poisson.nnx), np.ones(poisson.nny)
linear_x = np.linspace(0, 1.0, poisson.nnx)
linear_y = np.linspace(0, 1.0, poisson.nny)

rtol = 1e-10
atol = 1e-10

class TestRhs:
    def test_random(self):
        case_dir = os.path.join(base_dir, 'cases/dirichlet/random_left/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy').reshape(-1)
        potential = np.load(f'{case_dir}potential.npy')
        random_y = np.load(f'{case_dir}random_left.npy')
        poisson.solve(physical_rhs, zeros_x, zeros_x, random_y, zeros_y)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
    
    def test_constant_left(self):
        case_dir = os.path.join(base_dir, 'cases/dirichlet/constant_left/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy').reshape(-1)
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, zeros_x, zeros_x, ones_y, zeros_y)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
    
    def test_linear_pot_x(self):
        case_dir = os.path.join(base_dir, 'cases/dirichlet/linear_pot_x/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy').reshape(-1)
        potential = np.load(f'{case_dir}potential.npy')
        Vmax = 100.0
        poisson.solve(physical_rhs, Vmax * linear_x, 
                    Vmax * linear_x, zeros_y, Vmax * ones_y)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
