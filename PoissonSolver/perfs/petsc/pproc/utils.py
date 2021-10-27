import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt

default_cycler = (cycler(color=['darkblue', 'darkred', 'mediumblue', 'firebrick', 'royalblue', 'lightcoral']) +
                  cycler(linestyle=['-', '--', ':', '-.', '-', '--']))

plt.rc('lines', linewidth=1.8)
plt.rc('axes', prop_cycle=default_cycler)

def ax_prop(ax, xlabel, ylabel, scale='linear'):
    ax.grid(True)
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

def read_perfs(base_fn: str, nnxs: list):
    nnodes_list = list()
    best_times = list()
    av_times = list()
    stddev_times = list()

    # Read the elapsed times
    for nnx in nnxs:
        fp = open(f'{base_fn}/{nnx:d}.log', 'r')
        for line in fp:
            if '*------' in line:
                nnodes_list.append(int(fp.readline().strip('\n').split('=')[1]))
                fp.readline()
                best_times.append(float(fp.readline().strip('\n').split('=')[1]))
                av_times.append(float(fp.readline().strip('\n').split('=')[1]))
                stddev_times.append(float(fp.readline().strip('\n').split('=')[1]))
                break
        fp.close()

    nnodes_list = np.array(nnodes_list)
    best_times = np.array(best_times)
    av_times = np.array(av_times)
    stddev_times = np.array(stddev_times)

    return nnodes_list, best_times, av_times, stddev_times