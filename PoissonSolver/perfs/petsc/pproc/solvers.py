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
    fig_dir = Path('figures/solvers')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolution studied
    nnxs = [101, 201, 401, 801, 2001, 4001]

    # Plot precision vs time
    figname = 'perfs_36_procs'
    fig, ax = plt.subplots()
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/solvers/cg_gamg/rtol_1e-3/solver_cg_gamg_36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-3')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/solvers/cg_gamg/rtol_1e-6/solver_cg_gamg_36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-6')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/solvers/cg_gamg/rtol_1e-10/solver_cg_gamg_36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-10')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/solvers/direct_lu/solver_cg_gamg_36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='LU')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    ax_prop(ax, 'Number of nodes', 'Execution time [s]')
    fig.savefig(fig_dir / figname, bbox_inches='tight')

    # Plot precision vs time for hypre boomerang
    figname = 'perfs_36_procs_hypre'
    fig, ax = plt.subplots()
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/solvers/hypre_boomerang/rtol_1e-3/solver_cg_gamg_36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-3')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/solvers/hypre_boomerang/rtol_1e-6/solver_cg_gamg_36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-6')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/solvers/hypre_boomerang/rtol_1e-10/solver_cg_gamg_36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='rtol=1e-10')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    nnodes_list, best_times, av_times, stddev_times = read_perfs('../log/solvers/direct_lu/solver_cg_gamg_36_procs', nnxs)
    ax.plot(nnodes_list, av_times, label='LU')
    ax.fill_between(nnodes_list, av_times + stddev_times, av_times - stddev_times, alpha=.2)
    ax_prop(ax, 'Number of nodes', 'Execution time [s]')
    fig.savefig(fig_dir / figname, bbox_inches='tight')
