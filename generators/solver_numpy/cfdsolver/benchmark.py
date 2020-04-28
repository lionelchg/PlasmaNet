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
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main import main


def tts_vs_nnodes(config):
    """ Parametric study of the time to solution vs. number of nodes in the domain. """
    sizes = np.array([17, 33, 65, 101, 151, 201, 283, 501])
    times = np.zeros_like(sizes)

    for i, size in enumerate(tqdm(sizes)):
        config['mesh']['nnx'], config['mesh']['nny'] = size, size

        elapsed_time = time()
        main(config)
        elapsed_time = time() - elapsed_time
        times[i] = elapsed_time

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(sizes, times, 'o-')
    ax.set_xlabel('domain size')
    ax.set_ylabel('simulation duration')
    ax.set_title('Time to solution vs domain size over 100 iterations')
    fig.savefig('tts_vs_nnodes.png', dpi=150, bbox_inches='tight')


if __name__ == '__main__':

    with open('config.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    cfg['output']['save'] = 'none'
    cfg['output']['verbose'] = False

    # Simple time average
    ntests = 5
    total_time = 0
    for i in tqdm(range(ntests)):
        elapsed_time = time()
        main(cfg)
        elapsed_time = time() - elapsed_time
        total_time += elapsed_time

    print('\nAveraged elapsed time: {:.6f} s'.format(total_time / ntests))
    print('Time per iteration: {:.6e} it/s'.format(total_time / ntests / cfg['params']['nit']))

    # Parametric study vs number of nodes
    # tts_vs_nnodes(cfg)
