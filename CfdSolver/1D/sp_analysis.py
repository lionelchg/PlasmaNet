#!/Users/cheng/code/envs/dl/bin/python
import os
import numpy as np
import cmath
import matplotlib.pyplot as plt

fig_dir = 'figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

schemes = ['FOU', 'LW', 'SOU']

def G_LW(phi, sigma):
    """ Return the Lax-Wendroff scheme amplification factor """
    return 1 + sigma**2 * (np.cos(phi) - 1) - 1j * sigma * np.sin(phi)

def G_FOU(phi, sigma):
    """ Return the First Order Upwind scheme amplification factor """
    return 1 - sigma * (1 - np.exp(-1j * phi))

def G_SOU(phi, sigma):
    """ Return the Second Order Upwind scheme amplification factor """
    return (1 - sigma / 2 * (3 - 4 * np.exp(-1j * phi) + np.exp(-2j * phi))
                + sigma**2 / 2 * (1 - 2 * np.exp(-1j * phi) + np.exp(-2j * phi)))

def errors(G, phi, sigma):
    """ Computation of diffusion and dispersion error for a constant advection
    speed problem """
    G_num = G(phi, sigma)
    diff_err = abs(G_num)
    disp_err = np.zeros_like(phi)
    disp_err[0] = 1
    disp_err[1:] = np.array([- cmath.phase(G_num[i]) / sigma / phi[i] for i in range(1, len(phi))])
    return diff_err, disp_err

def plot_G(scheme, cfls):
    phi = np.linspace(0, np.pi, 300)
    phi_deg = phi * 180 / np.pi    
    fig, axes = plt.subplots(ncols=2, figsize=(10, 6))
    for cfl in cfls:
        df_err, dp_err = errors(amplf_dict[scheme], phi, cfl)
        axes[0].plot(phi_deg, df_err, label=f'CFL = {cfl:.2f}')
        axes[1].plot(phi_deg, dp_err, label=f'CFL = {cfl:.2f}')
    ax_prop(axes[0], r'$\varepsilon_D$')
    ax_prop(axes[1], r'$\varepsilon_\phi$')
    fig.suptitle(f'{scheme} Spectral Analysis')
    fig.savefig(fig_dir + f'errors_{scheme}', bbox_inches='tight')

def ax_prop(ax, ylabel, ylim=None):
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 180])

if __name__ == '__main__':
    amplf_dict = {}
    amplf_dict['FOU'] = G_FOU
    amplf_dict['LW'] = G_LW
    amplf_dict['SOU'] = G_SOU
    cfls = [0.25, 0.5, 0.8]
    for scheme in schemes:
        plot_G(scheme, cfls)