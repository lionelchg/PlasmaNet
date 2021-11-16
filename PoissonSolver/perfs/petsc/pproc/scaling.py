from pathlib import Path
from utils import plot_perfs

if __name__ == '__main__':
    # Figures directory
    fig_dir = Path('figures/scaling')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolution studied
    nnxs = [801, 2001, 4001]

    # Plotting resolution times for different number of CPU cores
    figname = 'perfs_scaling_cg_gamg_1e-10'
    log_fns = ['../log/cart/scaling/cg_gamg/9_procs/rtol_1e-10',
            '../log/cart/scaling/cg_gamg/18_procs/rtol_1e-10',
            '../log/cart/scaling/cg_gamg/36_procs/rtol_1e-10']
    labels = ['9 cores', '18 cores', '36 cores']
    plot_perfs(log_fns, labels, nnxs, 'linear', fig_dir / figname)
