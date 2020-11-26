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
from numpy.linalg import cond
import scipy.constants as co

from poissonsolver.poisson import Poisson
from poissonsolver.funcs import gaussian
from poissonsolver.utils import create_dir

def print_cond(sparse_mat, norm_list):
    mat = sparse_mat.todense()
    for norm in norm_list:
        print(f'K_{norm:.0f} = {cond(mat, norm):.2e}')

if __name__ == '__main__':
    norm_list = [1, np.inf]

    xmin, xmax = 0, 1.0
    ymin, ymax = 0, 1.0
    
    nnxs = [31, 51]
    for nnx in nnxs:
        print(f'nnx = {nnx:d}')
        nny = nnx
        poisson = Poisson(xmin, xmax, nnx, ymin, ymax, nny, 'cart_dirichlet')

        poisson_rx = Poisson(xmin, xmax, nnx, ymin, ymax, nny, 'axi_dirichlet')

        print('CDM K(A):')
        print_cond(poisson.mat, norm_list)

        print('\nADM K(A):')
        print_cond(poisson_rx.mat, norm_list)
