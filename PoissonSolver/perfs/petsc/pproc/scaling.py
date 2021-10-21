import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import ax_prop, read_perfs

if __name__ == '__main__':
    # Figures directory
    fig_dir = Path('figures/scaling')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolution studied
    nnxs = [101, 201, 401, 801, 2001, 4001]

    # Plot precision vs time
    figname = 'perfs_scaling_cg_gamg_1e-10'
    fig, ax = plt.subplots()
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/scaling/cg_gamg/9_procs/rtol_1e-10', nnxs)
    ax.plot(nnodes_list, av_times, label='9 procs')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/scaling/cg_gamg/18_procs/rtol_1e-10', nnxs)
    ax.plot(nnodes_list, av_times, label='18 procs')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/scaling/cg_gamg/36_procs/rtol_1e-10', nnxs)
    ax.plot(nnodes_list, av_times, label='36 procs')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    ax_prop(ax, 'Number of nodes', 'Execution time [s]')
    fig.savefig(fig_dir / figname, bbox_inches='tight')

