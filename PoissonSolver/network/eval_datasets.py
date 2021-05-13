############################################################################################################
#                                                                                                          #
#                                    Network evaluation on datasets                                        #
#                                                                                                          #
#                                  Lionel Cheng, CERFACS, 16.04.2021                                       #
#                                                                                                          #
############################################################################################################
import argparse
import yaml
from pathlib import Path

# From PlasmaNet
from PlasmaNet.poissonsolver.network import PoissonNetwork

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PoissonNetwork runs')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Config file path (default: None)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Neural network configuration
    config['network']['eval'] = config['eval']
    poisson_nn = PoissonNetwork(config['network'])

    # Evaluate on the datasets specified in config
    for ds_name, ds_loc in config['datasets'].items():
        poisson_nn.evaluate(ds_loc, Path(config['network']['casename']) / 'datasets' / ds_name)