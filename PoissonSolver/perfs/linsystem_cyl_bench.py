########################################################################################################################
#                                                                                                                      #
#                                         PoissonLinSystem benchmarks runner                                           #
#                                                                                                                      #
#                                         Lionel Cheng, CERFACS, 24.09.2021                                            #
#                                                                                                                      #
########################################################################################################################

import os
import yaml

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
from itertools import product
import copy

from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
import PlasmaNet.common.profiles as pf


if __name__ == '__main__':
    basecase_dir = 'cases/'

    # Read benchmark config file
    with open('bench_cyl_config.yml') as yaml_stream:
        bench_cfg = yaml.safe_load(yaml_stream)

    # Read PoissonLinSystem cylindrical config file
    with open('linsystem_cyl_config.yml') as yaml_stream:
        cfg_cyl = yaml.safe_load(yaml_stream)
    cfg_cart = copy.deepcopy(cfg_cyl)
    cfg_cart['geom'] = 'cartesian'
    cfg_cart['bcs'] = {'left': 'dirichlet', 'right': 'dirichlet',
                        'top': 'dirichlet', 'bottom': 'neumann'}

    # Iterate over sizes
    for (nnx, nny), solver_type, useUmfpack, assumeSortedIndices in product(
        zip(bench_cfg["nnx"], bench_cfg["nny"]),
        bench_cfg["solver_types"],
        bench_cfg["useUmfpack"],
        bench_cfg["assumeSortedIndices"]
    ):
        # Set sizes and initialize Poisson linear solver
        cfg_cart["nnx"], cfg_cart["nny"] = nnx, nny
        cfg_cart["solver_type"] = solver_type
        cfg_cart["solver_options"]["useUmfpack"] = useUmfpack
        cfg_cart["solver_options"]["assumeSortedIndices"] = assumeSortedIndices
        poisson = PoissonLinSystem(cfg_cart)
        zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
        pot_bcs = {'left': zeros_y, 'right': zeros_y, 'bottom': zeros_x, 'top': zeros_x}

        # Set sizes and initialize cylindrical Poisson linear solver
        cfg_cyl["nnx"], cfg_cyl["nny"] = nnx, nny
        cfg_cyl["solver_type"] = solver_type
        cfg_cyl["solver_options"]["useUmfpack"] = useUmfpack
        cfg_cyl["solver_options"]["assumeSortedIndices"] = assumeSortedIndices
        poisson_cyl = PoissonLinSystem(cfg_cyl)
        pot_bcs_cyl = {'left': zeros_y, 'right': zeros_y, 'top': zeros_x}

        # Create a solution
        ni0 = 1e16
        sigma_x, sigma_y = 5e-4, 5e-4
        x0, y0 = 2e-3, 0.0
        physical_rhs = pf.gaussian(poisson.X, poisson.Y, ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
        case_desc = '_'.join([str(tmp) for tmp in [nnx, nny, solver_type, useUmfpack, assumeSortedIndices]])
        print("------------------------------------------------------")
        print(f"{case_desc}")
        print("------------------------------------------------------")
        case_dir = f'{basecase_dir}{case_desc}/'

        # Solve Poisson equation with plotting and saving deactivated
        for i in range(bench_cfg["nits"]):
            print(f"{case_desc}_cart_", end="")
            poisson.run_case(f'{case_dir}cart', physical_rhs, pot_bcs, save=False, plot=(i==0))
            print(f"{case_desc}_cyl_", end="")
            poisson_cyl.run_case(f'{case_dir}cyl', physical_rhs, pot_bcs_cyl, save=False, plot=(i==0))
