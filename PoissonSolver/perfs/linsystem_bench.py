########################################################################################################################
#                                                                                                                      #
#                                         PoissonLinSystem benchmarks runner                                           #
#                                                                                                                      #
#                                      Guillaume Bogopolsky, CERFACS, 28.06.2021                                       #
#                                                                                                                      #
########################################################################################################################

import os
import yaml

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
from itertools import product

from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
import PlasmaNet.common.profiles as pf


if __name__ == '__main__':
    basecase_dir = f'{os.getenv("POISSON_DIR")}/cases/'

    # Read benchmark config file
    with open('bench_config.yml') as yaml_stream:
        bench_cfg = yaml.safe_load(yaml_stream)

    # Read PoissonLinSystem config file
    with open('linsystem_config.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    # Iterate over sizes
    for nn, solver_type, useUmfpack, assumeSortedIndices in product(
            bench_cfg["sizes"],
            bench_cfg["solver_types"],
            bench_cfg["useUmfpack"],
            bench_cfg["assumeSortedIndices"]
    ):
        # Set sizes and initialize Poisson linear solver
        cfg["nnx"], cfg["nny"] = nn, nn
        cfg["solver_type"] = solver_type
        cfg["solver_options"]["useUmfpack"] = useUmfpack
        cfg["solver_options"]["assumeSortedIndices"] = assumeSortedIndices
        poisson = PoissonLinSystem(cfg)
        zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
        pot_bcs = {'left': zeros_y, 'right': zeros_y, 'bottom': zeros_x, 'top': zeros_x}

        # Create a solution
        ni0 = 1e11
        sigma_x, sigma_y = 1e-3, 1e-3
        x0, y0 = 0.6e-2, 0.5e-2
        x01, y01 = 0.4e-2, 0.5e-2
        physical_rhs = pf.two_gaussians(poisson.X, poisson.Y, ni0, x0, y0, sigma_x, sigma_y,
                                        x01, y01, sigma_x, sigma_y) * co.e / co.epsilon_0
        case_desc = '_'.join([str(tmp) for tmp in [nn, solver_type, useUmfpack, assumeSortedIndices]])
        print("------------------------------------------------------")
        print(f"{case_desc}")
        print("------------------------------------------------------")
        case_dir = f'{basecase_dir}{case_desc}/'

        # Solve Poisson equation with plotting and saving deactivated
        for i in range(bench_cfg["nits"]):
            print(f"{case_desc}_", end="")
            poisson.run_case(case_dir, physical_rhs, pot_bcs, save=False, plot=False)
