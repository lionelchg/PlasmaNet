import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from pathlib import Path

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

def plot_perfs(log_fns: list, labels: str, nnxs: list, scale:str,
        figname: Path):
    """ Performance plot for a given number of log filenames and labels """
    fig, ax = plt.subplots(figsize=(5, 5))
    for log_fn, label in zip(log_fns, labels):
        nnodes_list, _, av_times, _ = read_perfs(log_fn, nnxs)
        ax.plot(nnodes_list, av_times, label=label)
    ax_prop(ax, 'Number of mesh nodes', 'Execution time [s]', scale)
    fig.savefig(figname, bbox_inches='tight')