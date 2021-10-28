import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import ax_prop, read_perfs

def plot_perfs(log_fns: list, labels: str, nnxs: list, scale:str,
        figname: Path):
    """ Performance plot for a given number of log filenames and labels """
    fig, ax = plt.subplots(figsize=(5, 5))
    for log_fn, label in zip(log_fns, labels):
        nnodes_list, _, av_times, _ = read_perfs(log_fn, nnxs)
        ax.plot(nnodes_list, av_times, label=label)
    ax_prop(ax, 'Number of mesh nodes', 'Execution time [s]', scale)
    fig.savefig(figname, bbox_inches='tight')

if __name__ == '__main__':
    # Figures directory
    fig_dir = Path('figures/solvers')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolution studied
    nnxs = [101, 201, 401, 801, 2001, 4001, 5001, 5501, 6001]

    # CG-GAMG
    log_fn_base = '../log/cart/solvers/cg_gamg/128_procs/'
    log_fns = ['rtol_1e-3', 'rtol_1e-6', 'rtol_1e-10', 'rtol_1e-12']
    log_fns = [log_fn_base + tmp for tmp in log_fns]
    labels = ['rtol=1e-3', 'rtol=1e-6', 'rtol=1e-10', 'rtol=1e-12']
    figname = 'perfs_128_procs_cg_gamg_linear'
    plot_perfs(log_fns, labels, nnxs, 'linear', fig_dir / figname)
    figname = 'perfs_128_procs_cg_gamg_log'
    plot_perfs(log_fns, labels, nnxs, 'log', fig_dir / figname)

    # CG-BoomerAMG
    log_fn_base = '../log/cart/solvers/hypre_boomeramg/128_procs/'
    log_fns = ['rtol_1e-3', 'rtol_1e-6', 'rtol_1e-10', 'rtol_1e-12']
    log_fns = [log_fn_base + tmp for tmp in log_fns]
    labels = ['rtol=1e-3', 'rtol=1e-6', 'rtol=1e-10', 'rtol=1e-12']
    figname = 'perfs_128_procs_hypre_boomeramg_linear'
    plot_perfs(log_fns, labels, nnxs, 'linear', fig_dir / figname)
    figname = 'perfs_128_procs_hypre_boomeramg_log'
    plot_perfs(log_fns, labels, nnxs, 'log', fig_dir / figname)

    # Combined plots for CG-GAMG and CG-BoomerAMG
    log_fns = ['../log/cart/solvers/cg_gamg/128_procs/rtol_1e-3',
            '../log/cart/solvers/cg_gamg/128_procs/rtol_1e-12',
            '../log/cart/solvers/hypre_boomeramg/128_procs/rtol_1e-3',
            '../log/cart/solvers/hypre_boomeramg/128_procs/rtol_1e-12']
    labels = [r'rtol = $10^{-3}$ GAMG', r'rtol = $10^{-12}$ GAMG',
        r'rtol = $10^{-3}$ BoomerAMG', r'rtol = $10^{-12}$ BoomerAMG']
    figname = 'perfs_128_procs_iterative_linear'
    plot_perfs(log_fns, labels, nnxs, 'linear', fig_dir / figname)
    figname = 'perfs_128_procs_iterative_log'
    plot_perfs(log_fns, labels, nnxs, 'log', fig_dir / figname)