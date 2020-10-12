########################################################################################################################
#                                                                                                                      #
#                                               Scalar residual schemes                                                #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from numba import njit


@njit(cache=True)
def compute_flux(res, a, u, diff_flux, sij, ncx, ncy, r=None):
    """ Iterate over cells (their edges) to compute the flux and residual. """
    for i in range(ncx):
        for j in range(ncy):
            if r is not None:
                edge_flux(res, a, u, diff_flux, sij * (0.75 * r[j] + 0.25 * r[j + 1]), i, j, i + 1, j, 0)
                edge_flux(res, a, u, diff_flux, sij * (0.25 * r[j] + 0.75 * r[j + 1]), i, j + 1, i + 1, j + 1, 0)
                edge_flux(res, a, u, diff_flux, sij * (r[j] + r[j + 1]) / 2, i, j, i, j + 1, 1)
                edge_flux(res, a, u, diff_flux, sij * (r[j] + r[j + 1]) / 2, i + 1, j, i + 1, j + 1, 1)
            else:
                edge_flux(res, a, u, diff_flux, sij, i, j, i + 1, j, 0)
                edge_flux(res, a, u, diff_flux, sij, i, j + 1, i + 1, j + 1, 0)
                edge_flux(res, a, u, diff_flux, sij, i, j, i, j + 1, 1)
                edge_flux(res, a, u, diff_flux, sij, i + 1, j, i + 1, j + 1, 1)


@njit(cache=True)
def edge_flux(res, a, u, diff_flux, sij, i1, j1, i2, j2, dim):
    """ Convection-diffusion flux. Implemented with 1st order upwind scheme for convection
    and second order centered scheme for diffusion """
    # Convective flux
    scalar_product = 0.5 * (a[dim, j1, i1] + a[dim, j2, i2]) * sij[dim]
    if scalar_product >= 0:
        flux = scalar_product * u[j1, i1]
    else:
        flux = scalar_product * u[j2, i2]
    # Diffusive flux
    flux -= 0.5 * (diff_flux[dim, j1, i1] + diff_flux[dim, j2, i2]) * sij[dim]
    res[j1, i1] += flux
    res[j2, i2] -= flux
