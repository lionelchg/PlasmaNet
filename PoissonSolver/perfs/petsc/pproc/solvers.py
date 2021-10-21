import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import ax_prop, read_perfs

if __name__ == '__main__':
    # Figures directory
    fig_dir = Path('figures/solvers')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolution studied
    nnxs = [101, 201, 401, 801, 2001, 4001]

    # Plot precision vs time
    figname = 'perfs_36_procs_cg_gamg'
    fig, ax = plt.subplots()
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/solvers/cg_gamg/36_procs/rtol_1e-3', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-3')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/solvers/cg_gamg/36_procs/rtol_1e-6', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-6')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/solvers/cg_gamg/36_procs/rtol_1e-10', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-10')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    ax_prop(ax, 'Number of nodes', 'Execution time [s]')
    fig.savefig(fig_dir / figname, bbox_inches='tight')

    # Plot precision vs time
    figname = 'perfs_36_procs_cg_hypre_boomerang'
    fig, ax = plt.subplots()
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/solvers/hypre_boomerang/36_procs/rtol_1e-3', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-3')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/solvers/hypre_boomerang/36_procs/rtol_1e-6', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-6')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/solvers/hypre_boomerang/36_procs/rtol_1e-10', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-10')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    ax_prop(ax, 'Number of nodes', 'Execution time [s]')
    fig.savefig(fig_dir / figname, bbox_inches='tight')

    # Plot precision vs time
    figname = 'perfs_36_procs_direct_solvers'
    fig, ax = plt.subplots()
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/solvers/direct_lu/36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='LU')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/cart/solvers/direct_cholesky/36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='Cholesky')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    ax_prop(ax, 'Number of nodes', 'Execution time [s]')
    fig.savefig(fig_dir / figname, bbox_inches='tight')

