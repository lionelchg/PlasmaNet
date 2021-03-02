########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co

from PlasmaNet.common.utils import create_dir
from PlasmaNet.poissonsolver.poisson import Poisson
import PlasmaNet.common.profiles as pf

xmin, xmax, nnx = 0, 0.01, 101
ymin, ymax, nny = 0, 0.01, 101

zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)

poisson = Poisson(xmin, xmax, nnx, ymin, ymax, nny, 'cart_dirichlet', 15)

rtol = 1e-10
atol = 1e-10

class TestRhs:
    def test_gaussian(self):
        case_dir = 'cases/rhs/gaussian/'
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy').reshape(-1)
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
    
    def test_step(self):
        case_dir = 'cases/rhs/step/'
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy').reshape(-1)
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
    
    def test_two_gaussians(self):
        case_dir = 'cases/rhs/two_gaussians/'
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy').reshape(-1)
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)

    def test_random_2D(self):
        case_dir = 'cases/rhs/random_2D/'
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy').reshape(-1)
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)

    def test_sin_2D(self):
        case_dir = 'cases/rhs/sin_2D/'
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy').reshape(-1)
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)