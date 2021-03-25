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

cfg['bcs'] = 'perio'
poisson = PoissonLinSystem(cfg)

zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)

rtol = 1e-10
atol = 1e-10

bcs = {'perio'}
A_x4_y3 = np.array([[-4, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                    [ 1,-4, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [ 0, 1,-4, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                    [ 1, 0, 1,-4, 0, 0, 0, 1, 0, 0, 0, 1],
                    [ 1, 0, 0, 0,-4, 1, 0, 1, 1, 0, 0, 0],
                    [ 0, 1, 0, 0, 1,-4, 1, 0, 0, 1, 0, 0],
                    [ 0, 0, 1, 0, 0, 1,-4, 1, 0, 0, 1, 0],
                    [ 0, 0, 0, 1, 1, 0, 1,-4, 0, 0, 0, 1],
                    [ 1, 0, 0, 0, 1, 0, 0, 0,-4, 1, 0, 1],
                    [ 0, 1, 0, 0, 0, 1, 0, 0, 1,-4, 1, 0],
                    [ 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,-4, 1],
                    [ 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1,-4]])

A_x3_y4 = np.array([[-4, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [ 1,-4, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [ 1, 1,-4, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [ 1, 0, 0,-4, 1, 1, 1, 0, 0, 0, 0, 0],
                    [ 0, 1, 0, 1,-4, 1, 0, 1, 0, 0, 0, 0],
                    [ 0, 0, 1, 1, 1,-4, 0, 0, 1, 0, 0, 0],
                    [ 0, 0, 0, 1, 0, 0,-4, 1, 1, 1, 0, 0],
                    [ 0, 0, 0, 0, 1, 0, 1,-4, 1, 0, 1, 0],
                    [ 0, 0, 0, 0, 0, 1, 1, 1,-4, 0, 0, 1],
                    [ 1, 0, 0, 0, 0, 0, 1, 0, 0,-4, 1, 1],
                    [ 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,-4, 1],
                    [ 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,-4]])

class TestRhs:

    def test_simple_x4_y3(self):
        cfg['bcs'] = 'perio'
        cfg['xmin'] =  0.0
        cfg['xmax'] =  3
        cfg['nnx'] =  4
        cfg['ymin'] =  0.0
        cfg['ymax'] =  2
        cfg['nny'] =  3
        poisson = PoissonLinSystem(cfg)
        Aperio = poisson.mat.toarray()
        assert np.allclose(Aperio, A_x4_y3, atol=atol, rtol=rtol)

    def test_simple_x3_y4(self):
        cfg['bcs'] = 'perio'
        cfg['xmin'] =  0.0
        cfg['xmax'] =  2
        cfg['nnx'] =  3
        cfg['ymin'] =  0.0
        cfg['ymax'] =  3
        cfg['nny'] =  4
        poisson = PoissonLinSystem(cfg)
        Aperio = poisson.mat.toarray()
        assert np.allclose(Aperio, A_x3_y4, atol=atol, rtol=rtol)

