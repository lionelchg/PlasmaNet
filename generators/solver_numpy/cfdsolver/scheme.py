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
def compute_flux(res, a, u, diff_flux, sij, ncx, ncy):
    """ Iterate over cells (their edges) to compute the flux and residual. """
    for i in range(ncx):
        for j in range(ncy):
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


@njit(nogil=True, cache=True)
def compute_flux2(res, a, u, diff_flux, sij, ncx, ncy):
    """ Iterate over cells (their edges) to compute the flux and residual. """
    for i in range(ncx):
        for j in range(ncy):
            # Joining similar snippets (less cache misses ?)
            # Diffusive flux
            scalar_product0 = 0.5 * (a[0, j, i] + a[0, j, i + 1]) * sij[0]
            scalar_product1 = 0.5 * (a[0, j + 1, i] + a[0, j + 1, i + 1]) * sij[0]
            scalar_product2 = 0.5 * (a[1, j, i] + a[1, j + 1, i]) * sij[1]
            scalar_product3 = 0.5 * (a[1, j, i + 1] + a[1, j + 1, i + 1]) * sij[1]

            # Test orientation
            if scalar_product0 >= 0:
                flux0 = scalar_product0 * u[j, i]
            else:
                flux0 = scalar_product0 * u[j, i + 1]

            if scalar_product1 >= 0:
                flux1 = scalar_product1 * u[j + 1, i]
            else:
                flux1 = scalar_product1 * u[j + 1, i + 1]

            if scalar_product2 >= 0:
                flux2 = scalar_product2 * u[j, i]
            else:
                flux2 = scalar_product2 * u[j + 1, i]

            if scalar_product3 >= 0:
                flux3 = scalar_product3 * u[j, i + 1]
            else:
                flux3 = scalar_product3 * u[j + 1, i + 1]

            # Diffusive flux
            flux0 -= 0.5 * (diff_flux[0, j, i] + diff_flux[0, j, i + 1]) * sij[0]
            flux1 -= 0.5 * (diff_flux[0, j + 1, i] + diff_flux[0, j + 1, i + 1]) * sij[0]
            flux2 -= 0.5 * (diff_flux[1, j, i] + diff_flux[1, j + 1, i]) * sij[1]
            flux3 -= 0.5 * (diff_flux[1, j, i + 1] + diff_flux[1, j + 1, i + 1]) * sij[1]

            # Update residual
            res[j, i] += flux0
            res[j, i + 1] -= flux0
            res[j + 1, i] += flux1
            res[j + 1, i + 1] -= flux1
            res[j, i] += flux2
            res[j + 1, i] -= flux2
            res[j, i + 1] += flux3
            res[j + 1, i + 1] -= flux3

            # 0 -- edge_flux(res, a, u, diff_flux, sij, i, j, i + 1, j, 0)
            # scalar_product0 = 0.5 * (a[0, j, i] + a[0, j, i + 1]) * sij[0]
            # if scalar_product0 >= 0:
            #     flux0 = scalar_product0 * u[j, i]
            # else:
            #     flux0 = scalar_product0 * u[j, i + 1]
            # Diffusive flux
            # flux0 -= 0.5 * (diff_flux[0, j, i] + diff_flux[0, j, i + 1]) * sij[0]
            # res[j, i] += flux0
            # res[j, i + 1] -= flux0

            # 1 -- edge_flux(res, a, u, diff_flux, sij, i, j + 1, i + 1, j + 1, 0)
            # scalar_product1 = 0.5 * (a[0, j + 1, i] + a[0, j + 1, i + 1]) * sij[0]
            # if scalar_product1 >= 0:
            #     flux1 = scalar_product1 * u[j + 1, i]
            # else:
            #     flux1 = scalar_product1 * u[j + 1, i + 1]
            # Diffusive flux1
            # flux1 -= 0.5 * (diff_flux[0, j + 1, i] + diff_flux[0, j + 1, i + 1]) * sij[0]
            # res[j + 1, i] += flux1
            # res[j + 1, i + 1] -= flux1

            # 2 -- edge_flux(res, a, u, diff_flux, sij, i, j, i, j + 1, 1)
            # scalar_product2 = 0.5 * (a[1, j, i] + a[1, j + 1, i]) * sij[1]
            # if scalar_product2 >= 0:
            #     flux2 = scalar_product2 * u[j, i]
            # else:
            #     flux2 = scalar_product2 * u[j + 1, i]
            # Diffusive flux
            # flux2 -= 0.5 * (diff_flux[1, j, i] + diff_flux[1, j + 1, i]) * sij[1]
            # res[j, i] += flux2
            # res[j + 1, i] -= flux2

            # 3 -- edge_flux(res, a, u, diff_flux, sij, i + 1, j, i + 1, j + 1, 1)
            # scalar_product3 = 0.5 * (a[1, j, i + 1] + a[1, j + 1, i + 1]) * sij[1]
            # if scalar_product3 >= 0:
            #     flux3 = scalar_product3 * u[j, i + 1]
            # else:
            #     flux3 = scalar_product3 * u[j + 1, i + 1]
            # Diffusive flux
            # flux3 -= 0.5 * (diff_flux[1, j, i + 1] + diff_flux[1, j + 1, i + 1]) * sij[1]
            # res[j, i + 1] += flux3
            # res[j + 1, i + 1] -= flux3
