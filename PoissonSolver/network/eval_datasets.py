############################################################################################################
#                                                                                                          #
#                                    Network evaluation on datasets                                        #
#                                                                                                          #
#                                  Lionel Cheng, CERFACS, 16.04.2021                                       #
#                                                                                                          #
############################################################################################################
import argparse
import yaml
import numpy as np
import scipy.constants as co

# From PlasmaNet
from PlasmaNet.poissonsolver.network import PoissonNetwork

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PoissonNetwork runs')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Neural network config_1/random_4
    config['network']['eval'] = config['eval']
    poisson_nn = PoissonNetwork(config['network'])
    poisson_nn.evaluate('/scratch/cfd/PlasmaDL/datasets/eval/101x101/stand_alone/random_8', 
                        'cases/eval')