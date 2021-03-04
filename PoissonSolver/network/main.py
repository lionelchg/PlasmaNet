import argparse
import yaml
import numpy as np
from PlasmaNet.poissonsolver.network import PoissonNetwork
from PlasmaNet.poissonsolver.linsystem import Poisson

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PoissonNetwork runs')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args.add_argument('-d', '--datadir', default=None, type=str,
                      help='Dataset directory (should be .npy)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Create neural network instance and linear system instance
    pnn = PoissonNetwork(config)
    pls = Poisson(pnn.xmin, pnn.xmax, pnn.nnx, pnn.ymin, pnn.ymax, pnn.nny, 
            'cart_dirichlet')
    physical_rhs = np.load(args.datadir)

    # Solve using neural network
    pnn.solve(physical_rhs, pnn.nnx, pnn.Lx, pnn.Ly)