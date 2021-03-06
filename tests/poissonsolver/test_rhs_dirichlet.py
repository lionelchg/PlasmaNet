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
cfg['bcs'] = 'dirichlet'

poisson = PoissonLinSystem(cfg)

zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)

rtol = 1e-10
atol = 1e-10

bcs = {'left':zeros_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}

class TestRhs:
    def test_gaussian(self):
        case_dir = os.path.join(base_dir, 'cases/dirichlet/rhs/gaussian/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy')
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, bcs)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
    
    def test_step(self):
        case_dir = os.path.join(base_dir, 'cases/dirichlet/rhs/step/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy')
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, bcs)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
    
    def test_two_gaussians(self):
        case_dir = os.path.join(base_dir, 'cases/dirichlet/rhs/two_gaussians/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy')
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, bcs)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)

    def test_random_2D(self):
        case_dir = os.path.join(base_dir, 'cases/dirichlet/rhs/random_2D/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy')
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, bcs)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)

    def test_sin_2D(self):
        case_dir = os.path.join(base_dir, 'cases/dirichlet/rhs/sin_2D/')
        physical_rhs = np.load(f'{case_dir}physical_rhs.npy')
        potential = np.load(f'{case_dir}potential.npy')
        poisson.solve(physical_rhs, bcs)
        assert np.allclose(poisson.potential, potential, atol=atol, rtol=rtol)
