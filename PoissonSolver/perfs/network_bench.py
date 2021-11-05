########################################################################################################################
#                                                                                                                      #
#                                          PoissonNetwork benchmarks runner                                            #
#                                                                                                                      #
#                                      Guillaume Bogopolsky, CERFACS, 28.06.2021                                       #
#                                                                                                                      #
########################################################################################################################

import os
import torch
import yaml
from pathlib import Path
import numpy as np
import scipy.constants as co
from itertools import product
import argparse

from PlasmaNet.poissonsolver.network import PoissonNetwork
import PlasmaNet.common.profiles as pf


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Network performance monitoring')
    parser.add_argument('-n', '--nnxs', type=int, nargs='+', default=None,
                help='The different resolutions studied')
    parser.add_argument('-cn', '--casename', type=str, default=None,
                help='Casename (where the results are stored)')
    args = parser.parse_args()

    # Read PoissonNetwork config file
    base_config = "network_base_config.yml"
    with open(base_config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)
    config["network"]["eval"] = config["eval"]

    # Read benchmark config file
    with open('bench_config.yml') as yaml_stream:
        bench_cfg = yaml.safe_load(yaml_stream)

    # Erase if specified in command line
    if args.nnxs is not None:
        bench_cfg["sizes"] = args.nnxs
    if args.casename is not None:
        config["network"]["casename"] = args.casename

    # Iterate over the networks to evaluate and the sizes
    base_casename = config["network"]["casename"]
    for net, nn in product(bench_cfg["networks"], bench_cfg["sizes"]):
        # Set the options
        config["network"]["eval"]["nnx"] = nn
        config["network"]["eval"]["nny"] = nn
        config["network"]["resume"] = bench_cfg["networks"][net]["resume"]
        config["network"]["arch"] = bench_cfg["networks"][net]["arch"]
        if "input_res" in config["network"]["arch"]["args"]:
            config["network"]["arch"]["args"]["input_res"] = nn
        config["network"]["casename"] = os.path.join(base_casename, str(net), str(nn))

        print("------------------------------------------------------")
        print(f"network: {net} - size: {nn}")
        print("------------------------------------------------------")

        poisson = PoissonNetwork(config["network"])
        ni0 = 1e11
        sigma_x, sigma_y = 1e-3, 1e-3
        x0, y0 = 0.6e-2, 0.5e-2
        x01, y01 = 0.4e-2, 0.5e-2
        physical_rhs = pf.two_gaussians(poisson.X, poisson.Y, ni0, x0, y0, sigma_x, sigma_y,
                                        x01, y01, sigma_x, sigma_y) * co.e / co.epsilon_0
        for i in range(bench_cfg["nits"]):
            poisson.run_case(Path(config['network']['casename']), physical_rhs, plot=False, save=False)

        # Clean GPU cache?
        torch.cuda.empty_cache()
