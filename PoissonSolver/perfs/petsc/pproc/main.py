import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import ax_prop, read_perfs

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