########################################################################################################################
#                                                                                                                      #
#                                          PoissonNetwork benchmarks runner                                            #
#                                                                                                                      #
#                                      Guillaume Bogopolsky, CERFACS, 28.06.2021                                       #
#                                                                                                                      #
########################################################################################################################

import os
import argparse
import yaml
from pathlib import Path
import numpy as np
import scipy.constants as co
from itertools import product

from PlasmaNet.poissonsolver.network import PoissonNetwork
import PlasmaNet.common.profiles as pf


if __name__ == "__main__":
    # Read PoissonNetwork config file
    base_config = "network_base_config.yml"
    with open(base_config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Read benchmark config file
    with open('bench_config.yml') as yaml_stream:
        bench_cfg = yaml.safe_load(yaml_stream)

    # Iterate over the networks to evaluate and the sizes
    base_casename = config["network"]["casename"]
    for net, nn in product(bench_cfg["networks"], bench_cfg["sizes"]):
        # Set the options
        config["eval"]["nnx"] = nn
        config["eval"]["nny"] = nn
        config["network"]["resume"] = bench_cfg["networks"][net]["resume"]
        config["network"]["arch"] = bench_cfg["networks"][net]["arch"]
        config["network"]["casename"] = os.join(base_casename, str(net), str(nn))

        poisson_nn = PoissonNetwork(config["network"])
        ni0 = 1e11
        sigma_x, sigma_y = 1e-3, 1e-3
        x0, y0 = 0.6e-2, 0.5e-2
        x01, y01 = 0.4e-2, 0.5e-2
        physical_rhs = pf.two_gaussians(poisson.X, poisson.Y, ni0, x0, y0, sigma_x, sigma_y,
                                        x01, y01, sigma_x, sigma_y) * co.e / co.epsilon_0
        poisson.run_case(case_dir, physical_rhs, plot=False)
