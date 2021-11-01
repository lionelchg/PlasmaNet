import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import ax_prop, read_perfs
from cycler import cycler

def plot_perfs(log_fns: list, labels: str, nnxs: list, scale:str,
        figname: Path):
    """ Performance plot for a given number of log filenames and labels """
    fig, ax = plt.subplots(figsize=(5, 5))
    for log_fn, label in zip(log_fns, labels):
        nnodes_list, _, av_times, _ = read_perfs(log_fn, nnxs)
        ax.plot(nnodes_list, av_times, label=label)
    ax_prop(ax, 'Number of mesh nodes', 'Execution time [s]', scale)
    fig.savefig(figname, bbox_inches='tight', format='pdf')

if __name__ == '__main__':
    # Figures directory
    fig_dir = Path('figures/solvers')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolution studied
    nnxs = [101, 201, 401, 801, 2001, 4001, 5001, 5501, 6001]

    # Combined plots for CG-GAMG and CG-BoomerAMG
    default_cycler = (cycler(color=['mediumblue', 'royalblue', 'darkblue', 'firebrick']) +
                  cycler(linestyle=['-', '--', ':', ':']))
    plt.rc('lines', linewidth=1.8)
    plt.rc('axes', prop_cycle=default_cycler)
    log_fns = ['../log/cart/solvers_rtol_1e-3/default/cg_boomeramg/36_procs',
            '../log/cart/solvers_rtol_1e-7/default/cg_boomeramg/36_procs',
            '../log/cart/solvers_rtol_1e-12/default/cg_boomeramg/36_procs']
    labels = [r'rtol= $10^{-3}$ BoomerAMG', r'rtol= $10^{-7}$ BoomerAMG',
                r'rtol= $10^{-12}$ BoomerAMG']
    figname = 'perfs_36_procs_iterative.pdf'
    plot_perfs(log_fns, labels, nnxs, 'linear', fig_dir / figname)

    # Combined plots for all solvers with GAMG at rtol=1e-3
    default_cycler = (cycler(color=['mediumblue', 'royalblue', 'darkblue', 'firebrick', 'darkred', 'lightcoral']) +
                  cycler(linestyle=['-', '--', ':', '-', '-.', '--']))
    plt.rc('lines', linewidth=1.8)
    plt.rc('axes', prop_cycle=default_cycler)
    log_fns = ['../log/cart/solvers_rtol_1e-3/default/cg_gamg/36_procs',
            '../log/cart/solvers_rtol_1e-3/default/cgs_gamg/36_procs',
            '../log/cart/solvers_rtol_1e-3/default/bcgs_gamg/36_procs',
            '../log/cart/solvers_rtol_1e-3/default/gmres_gamg/36_procs',
            '../log/cart/solvers_rtol_1e-3/default/minres_gamg/36_procs']
    labels = ['CG-GAMG', 'CGS-GAMG', 'BiCGStab-GAMG', 'GMRES-GAMG', 'MINRES-GAMG']
    figname = 'perfs_36_procs_rtol1e-3_solvers_gamg.pdf'
    plot_perfs(log_fns, labels, nnxs, 'linear', fig_dir / figname)

    # Combined plots for all solvers with GAMG at rtol=1e-3
    log_fns = ['../log/cart/solvers_rtol_1e-3/default/cg_boomeramg/36_procs',
            '../log/cart/solvers_rtol_1e-3/default/cgs_boomeramg/36_procs',
            '../log/cart/solvers_rtol_1e-3/default/bcgs_boomeramg/36_procs',
            '../log/cart/solvers_rtol_1e-3/default/gmres_boomeramg/36_procs',
            '../log/cart/solvers_rtol_1e-3/default/minres_boomeramg/36_procs']
    labels = ['CG-BoomerAMG', 'CGS-BoomerAMG', 'BiCGStab-BoomerAMG', 'GMRES-BoomerAMG', 'MINRES-BoomerAMG']
    figname = 'perfs_36_procs_rtol1e-3_solvers_boomeramg.pdf'
    plot_perfs(log_fns, labels, nnxs, 'linear', fig_dir / figname)