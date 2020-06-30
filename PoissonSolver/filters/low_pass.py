import os
import numpy as np
import matplotlib.pyplot as plt

fig_dir = 'figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def lp_forder(K, f):
    return K / (1 + f * 1j)

def lp_sorder(K, f, Q):
    return K / (1 - f**2 + f / Q * 1j)

def plot_filter(f, G, figname):
    fig, axes = plt.subplots(nrows=2, figsize=(6, 10))
    axes[0].plot(f, np.absolute(G))
    axes[0].grid(True)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title('Gain')
    axes[0].set_xlabel(r'$f/f_c$')
    axes[0].set_ylabel(r'$G(f/f_c)$')

    axes[1].plot(f, np.angle(G) * 180 / np.pi)
    axes[1].grid(True)
    axes[1].set_xscale('log')
    axes[1].set_title('Phase')
    axes[1].set_xlabel(r'$f/f_c$')
    axes[1].set_ylabel(r'$\phi(f/f_c)$ [deg]')

    plt.savefig(figname, bbox_inches='tight')

if __name__ == '__main__':
    # frequencies
    f = np.logspace(-2, 4, 500)
    # quality for second order
    Q = 4

    G_1st = lp_forder(1, f)
    G_2nd = lp_sorder(1, f, Q)

    plot_filter(f, G_1st, fig_dir + 'lp_forder')
    plot_filter(f, G_2nd, fig_dir + 'lp_sorder')