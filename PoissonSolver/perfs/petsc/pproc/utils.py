import numpy as np

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