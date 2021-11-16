########################################################################################################################
#                                                                                                                      #
#                                      Morrow chemistry implementation functions                                       #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 05.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import scipy.constants as co
import matplotlib.pyplot as plt
from numba import njit

# From current library
from ...poissonscreensolver.photo import photo_coeff, quenching_press
from ...common.utils import create_dir


@njit(cache=True)
def morrow(mu, D, E_field, nd, resnd, nnx, nny, voln, irate=None):
    """ Apply chemistry from computed coefficients. """
    ngas, Tgas = 2.688e25, 300
    beta_recombination = 2e-13
    Pgas = ngas * co.k * Tgas
    quenching_coef = quenching_press / (quenching_press + Pgas)

    for i in range(nnx):
        for j in range(nny):
            normE = np.sqrt(E_field[0, j, i]**2 + E_field[1, j, i]**2)
            mu[j, i] = mobility_coeff_morrow(normE, ngas)
            D[j, i] = diffusion_coeff_morrow(normE, ngas, mu[j, i])
            ionization_freq = ionization_rate_morrow(normE / ngas, ngas) * mu[j, i] * normE
            attachment_freq = attachment_rate_morrow(normE / ngas, ngas) * mu[j, i] * normE

            if irate is not None: irate[j, i] = nd[0, j, i] * ionization_freq * photo_coeff(normE / Pgas) * quenching_coef
            resnd[0, j, i] -= (nd[0, j, i] * ionization_freq - nd[0, j, i] * attachment_freq - nd[0, j, i] * nd[1, j, i] * beta_recombination) * voln[j, i]
            resnd[1, j, i] -= (nd[0, j, i] * ionization_freq - (nd[0, j, i] + nd[2, j, i]) * nd[1, j, i] * beta_recombination) * voln[j, i]
            resnd[2, j, i] -= (nd[0, j, i] * attachment_freq - nd[2, j, i] * nd[1, j, i] * beta_recombination) * voln[j, i]


@njit(cache=True)
def ionization_rate_morrow(E_N, N):
    """ Compute ionization rate from tabulated values. """
    E_N = E_N * 1e4
    if E_N > 1.5e-15:
        rate = N * 1e-6 * 2.0e-16 * np.exp(-7.248e-15 / E_N) * 1e2
    else:
        rate = N * 1e-6 * 6.619e-17 * np.exp(-5.593e-15 / E_N) * 1e2

    return rate


@njit(cache=True)
def attachment_rate_morrow(E_N, N):
    """ Compute attachment rate from tabulated values. """
    E_N = E_N * 1e4
    if E_N > 1.05e-15:
        rate = (N*1e-6) * (8.889e-5*E_N+2.567e-19) * 1e2
    else:
        rate = (N*1e-6) * (6.089e-4*E_N-2.893e-19) * 1e2

    if rate < 0:
        rate = 0e0

    if E_N > 1.0e-19:
        rate = rate + (N*1e-6)**2 * 4.7778e-59 * E_N**(-1.2749) * 1e2
    else:
        rate = rate + (N*1e-6)**2 * 4.7778e-59 * (1e-19)**(-1.2749) * 1e2

    return rate


@njit(cache=True)
def mobility_coeff_morrow(Eprim, N):
    """ Compute mobility coefficient from tabulated values. """
    E_N = Eprim / N * 1e4

    if E_N > 2.0e-15:
        mobility = (7.4e21*E_N + 7.1e6) / (Eprim * 1e-2)

    elif 1e-16 < E_N <= 2.0e-15:
        mobility = (1.03e22*E_N + 1.3e6) / (Eprim * 1e-2)

    elif 2.6e-17 < E_N <= 1e-16:
        mobility = (7.2973e21*E_N + 1.63e6) / (Eprim * 1e-2)

    elif E_N > 1e-19:
        mobility = (6.87e22*E_N + 3.38e4) / (Eprim * 1e-2)

    else:
        mobility = (6.87e22*1e-19 + 3.38e4) / ((1e-19*N/1e4) * 1e-2)

    mobility = mobility * 1.0e-4

    return mobility


@njit(cache=True)
def diffusion_coeff_morrow(Eprim, N, mobility):
    """ Compute diffusion coefficient from tabulated values. """
    E_N = Eprim / N * 1e4

    if E_N > 1e-19:
        coeff = 0.3341e9 * E_N**0.54069 * mobility
    else:
        coeff = 0.3341e9 * 1e-19**0.54069 * mobility

    return coeff


if __name__ == '__main__':
    fig_dir = 'figures/'
    create_dir(fig_dir)

    npoints = 201
    Elog = np.logspace(1, 7, npoints)
    Elin = np.linspace(1e5, 1e7, npoints)

    P, Tgas = co.atm, 300
    Ngas = P / co.k / Tgas

    mobility, diffusion, ioniz_freq, att_freq = \
        np.zeros_like(Elog), np.zeros_like(Elog), np.zeros_like(Elin), np.zeros_like(Elin)

    for i in range(npoints):
        mobility[i] = mobility_coeff_morrow(Elog[i], Ngas)
        diffusion[i] = diffusion_coeff_morrow(Elog[i], Ngas, mobility[i])
        ioniz_freq[i] = ionization_rate_morrow(Elin[i] / Ngas, Ngas) * mobility_coeff_morrow(Elin[i], Ngas) * Elin[i]
        att_freq[i] = attachment_rate_morrow(Elin[i] / Ngas, Ngas) * mobility_coeff_morrow(Elin[i], Ngas) * Elin[i]


    def ax_prop(ax, logax, title, xlabel, ylabel, ylim=None):
        """ Set axes configuration for plots. """
        ax.grid(True)
        ax.set_title(title)
        if logax == 'x':
            ax.set_xscale('log')
        elif logax == 'y':
            ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if ylim is not None:
            ax.set_ylim(ylim)


    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))

    axes[0][0].plot(Elog, mobility)
    ax_prop(axes[0][0], 'x', 'Mobility', '$E$ [V.m-1]', r'$\mu$ [m2.V-1.s-1]')
    axes[1][0].plot(Elog, diffusion)
    ax_prop(axes[1][0], 'x', 'Diffusion', '$E$ [V.m-1]', '$D$ [m2.s-1]')
    axes[0][1].plot(Elin, ioniz_freq)
    ax_prop(axes[0][1], 'y', 'Ionization Frequency', '$E$ [V.m-1]', r'$\nu$ [Hz]', ylim=[1e6, 1e11])
    axes[1][1].plot(Elin, att_freq)
    ax_prop(axes[1][1], 'y', 'Attachment Frequency', '$E$ [V.m-1]$', r'$\nu$ [Hz]', ylim=[1e6, 1e11])

    plt.savefig(fig_dir + 'morrow', bbox_inches='tight')
