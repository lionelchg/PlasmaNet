import numpy as np
from numba import njit


@njit(cache=True)
def morrow(mu, D, E_field, ne, rese, nionp, resp, nn, resn, nnx, nny):
    ngas = 2.688e25
    beta_recombination = 2e-13
    ionization_rate, attachment_rate = 0, 0

    for i in range(nnx):
        for j in range(nny):
            normE = np.sqrt(E_field[0, j, i]**2 + E_field[1, j, i]**2)
            ionization_rate_morrow(normE / ngas * 1e4, ionization_rate, ngas)
            attachment_rate_morrow(normE / ngas * 1e4, attachment_rate, ngas)
            mobility_coeff_morrow(normE, ngas, mu[j, i])
            diffusion_coeff_morrow(normE, ngas, D[j, i], mu[j, i])

            resp[j, i] -= ne[j, i] * ionization_rate * mu[j, i] * normE \
              - (ne[j, i] + nn[j, i]) * nionp[j, i] * beta_recombination
            resn[j, i] -= ne[j, i] * attachment_rate * mu[j, i] * normE \
            - nn[j, i] * nionp[j, i] * beta_recombination
            rese[j, i] -= ne[j, i] * ionization_rate * mu[j, i] * normE \
            - ne[j, i] * attachment_rate * mu[j, i] * normE \
             - ne[j, i] * nionp[j, i] * beta_recombination


@njit(cache=True)
def ionization_rate_morrow(E_N, rate, N):
    if (E_N > 1.5e-15):
        rate = (N*1e-6) * 2.0e-16*np.exp(-7.248e-15 / E_N) * 1e2
    else:
        rate = (N*1e-6) * 6.619e-17 * np.exp(-5.593e-15 / E_N) * 1e2


@njit(cache=True)
def attachment_rate_morrow(E_N, rate, N):
    if (E_N > 1.05e-15):
        rate = (N*1e-6) * (8.889e-5*E_N+2.567e-19) * 1e2
    else:
        rate = (N*1e-6) * (6.089e-4*E_N-2.893e-19) * 1e2

    if (rate < 0):
        rate = 0e0

    if (E_N > 1.0e-19):
        rate = rate + (N*1e-6)**2 * 4.7778e-59 * E_N**(-1.2749) * 1e2
    else:
        rate = rate + (N*1e-6)**2 * 4.7778e-59 * (1e-19)**(-1.2749) * 1e2


@njit(cache=True)
def mobility_coeff_morrow(Eprim, N, mobility):
    E_N = Eprim / N * 1e4

    if (E_N > 2.0e-15):
        
        mobility = (7.4e21*E_N + 7.1e6) / (Eprim * 1e-2)

    elif (E_N > 1e-16 and E_N <= 2.0e-15):

        mobility = (1.03e22*E_N + 1.3e6) / (Eprim * 1e-2)

    elif (E_N > 2.6e-17 and E_N <= 1e-16):

        mobility = (7.2973e21*E_N + 1.63e6) / (Eprim * 1e-2)

    elif (E_N > 1e-19 ):

        mobility = (6.87e22*E_N + 3.38e4) / (Eprim * 1e-2)

    else:

        mobility = (6.87e22*1e-19 + 3.38e4) / ((1e-19*N/1e4) * 1e-2)

    mobility = mobility * 1.0e-4


@njit(cache=True)
def diffusion_coeff_morrow(Eprim, N, coeff, mobility):
    E_N = Eprim / N * 1e4

    if (E_N > 1e-19):
        coeff = 0.3341e9 * E_N**0.54069 * mobility
    else:
        coeff = 0.3341e9 * 1e-19**0.54069 * mobility
