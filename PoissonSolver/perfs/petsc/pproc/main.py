import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ax_prop(ax, xlabel, ylabel):
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_yscale('log')
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
        fp = open(f'{base_fn}_{nnx:d}.log', 'r')
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

if __name__ == '__main__':
    # Figures directory
    fig_dir = Path('figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    figname = 'perfs_rome_128'

    # Resolution studied
    nnxs = [101, 201, 401, 801, 2001, 4001, 5001]

    # Plot the read values
    fig, ax = plt.subplots()
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/A100/solver_cg_gamg_128_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='A100')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/V100/solver_cg_gamg_36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='V100')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    ax_prop(ax, 'Number of nodes', 'Execution time [s]')
    fig.savefig(fig_dir / figname, bbox_inches='tight')