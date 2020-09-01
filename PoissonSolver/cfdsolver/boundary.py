########################################################################################################################
#                                                                                                                      #
#                                            Boundary conditions functions                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################

import copy


def outlet_y(res, a, u, diff_flux, dx, yb, r=None):
    """ Outlet boundary conditions in the y direction """
    flux = dx / 2 * (0.75 * (a[1, yb, 1:] * u[yb, 1:] + diff_flux[1, yb, 1:]) + 
                     0.25 * (a[1, yb, :-1] * u[yb, :-1] + diff_flux[1, yb, :-1]))
    if r is not None:
        res[yb, 1:] += flux * r
    else:
        res[yb, 1:] += flux
    flux = dx / 2 * (0.25 * (a[1, yb, 1:] * u[yb, 1:] + diff_flux[1, yb, 1:]) + 
                     0.75 * (a[1, yb, :-1] * u[yb, :-1] + diff_flux[1, yb, :-1]))

    if r is not None:
        res[yb, :-1] += flux * r
    else:
        res[yb, :-1] += flux

    # res[yb, 1:-1] += dx * a[1, yb, 1:-1] * u[yb, 1:-1]
    # res[yb, 0] += dx / 2 * a[1, yb, 0] * u[yb, 0]
    # res[yb, -1] += dx / 2 * a[1, yb, -1] * u[yb, -1]


def outlet_x(res, a, u, diff_flux, dy, xb, r=None):
    """ Outlet boundary condition in the x direction """
    if r is not None:
        flux = dy / 2 * (0.75 * (a[0, 1:, xb] * u[1:, xb] + diff_flux[1, 1:, xb]) * r[1:, xb] + 
                         0.25 * (a[0, :-1, xb] * u[:-1, xb] + diff_flux[1, :-1, xb]) * r[:-1, xb])
        res[1:, xb] += flux
        flux = dy / 2 * (0.25 * (a[0, 1:, xb] * u[1:, xb] + diff_flux[1, 1:, xb]) * r[1:, xb] + 
                         0.75 * (a[0, :-1, xb] * u[:-1, xb] + diff_flux[1, :-1, xb]) * r[:-1, xb])
        res[:-1, xb] += flux
    else:
        flux = dy / 2 * (0.75 * (a[0, 1:, xb] * u[1:, xb] + diff_flux[1, 1:, xb]) + 
                         0.25 * (a[0, :-1, xb] * u[:-1, xb] + diff_flux[1, :-1, xb]))
        res[1:, xb] += flux
        flux = dy / 2 * (0.25 * (a[0, 1:, xb] * u[1:, xb] + diff_flux[1, 1:, xb]) + 
                         0.75 * (a[0, :-1, xb] * u[:-1, xb] + diff_flux[1, :-1, xb]))
        res[:-1, xb] += flux
    # res[1:-1, xb] += dy * a[0, 1:-1, xb] * u[1:-1, xb]
    # res[0, xb] += dy / 2 * a[0, 0, xb] * u[0, xb]
    # res[-1, xb] += dy / 2 * a[0, -1, xb] * u[-1, xb]

def perio_x(res):
    """ Periodic condition in the x direction """
    res[1:-1, 0] += res[1:-1, -1]
    res[1:-1, -1] = copy.deepcopy(res[1:-1, 0])


def perio_y(res):
    """ Periodic condition in the y direction """
    res[0, 1:-1] += res[-1, 1:-1]
    res[-1, 1:-1] = copy.deepcopy(res[0, 1:-1])


def full_perio(res):
    """ Full periodic conditions in the 4 directions """
    # Corner (only for full periodic)
    res[0, 0] = res[0, 0] + res[0, -1] + res[-1, 0] + res[-1, -1]
    res[0, -1], res[-1, 0], res[-1, -1] = res[0, 0], res[0, 0], res[0, 0]

    # Periodic boundary conditions - Sides
    res[1:-1, 0] += res[1:-1, -1]
    res[1:-1, -1] = copy.deepcopy(res[1:-1, 0])
    res[0, 1:-1] += res[-1, 1:-1]
    res[-1, 1:-1] = copy.deepcopy(res[0, 1:-1])
