########################################################################################################################
#                                                                                                                      #
#                                                Benchmarking routines                                                 #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################

import yaml
from time import time
from tqdm import tqdm

from main import main


if __name__ == '__main__':

    with open('config.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    cfg['output']['save'] = 'none'

    ntests = 5
    total_time = 0
    for i in tqdm(range(ntests)):
        elapsed_time = time()
        main(cfg)
        elapsed_time = time() - elapsed_time
        total_time += elapsed_time

    print('\nAveraged elapsed time: {:.6f} s'.format(total_time / ntests))
    print('Time per iteration: {:.6e} it/s'.format(total_time / ntests / cfg['params']['nit']))
