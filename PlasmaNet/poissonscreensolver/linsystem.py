########################################################################################################################
#                                                                                                                      #
#                            Routines concerning the linear system for Poisson screening equation                      #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from scipy import sparse
from ..poissonsolver.linsystem import dirichlet_line, neumann_line

def cart_matrix(dx: float, dy: float, nx: int, ny: int, coeff: float, scale: float, bcs: dict) -> sparse.csc_matrix:
    """ Generate a cartesian matrix with prescribed boundary conditions
    (Dirichlet or Neumann mixed) for the Screened Poisson equation

    :param dx: x resolution
    :type dx: float
    :param dy: y resolution
    :type dy: float
    :param nx: number of nodes in x direction
    :type nx: int
    :param ny: number of nodes in y direction
    :type ny: int
    :param coeff: coefficient in front of the diagonal part of the matrix
    :type coeff: float
    :param scale: scaling factor for the matrix coefficients
    :type scale: float
    :param bcs: boundary conditions (either string or list of strings)
    :type bcs: dict
    :return: matrix of the Poisson equation
    :rtype: sparse.csc_matrix
    """
    diags = np.zeros((5, nx * ny))
    diags[0, :] = - coeff * scale

    # Filling the diagonals, first the down neumann bc, then the dirichlet bc and finally the interior nodes
    for i in range(nx * ny):
        dc_bool = False
        # Start by filling in x direction
        if i % nx == 0:
            if bcs['left'] == 'dirichlet':
                dirichlet_line(diags, i, nx, ny)
                dc_bool = True
            elif bcs['left'] == 'neumann':
                neumann_line(diags, (i, (1, i + 1)), dx, scale)
        elif i % nx == nx - 1:
            if bcs['right'] == 'dirichlet':
                dirichlet_line(diags, i, nx, ny)
                dc_bool = True
            elif bcs['right'] == 'neumann':
                neumann_line(diags, (i, (2, i - 1)), dx, scale)
        else:
            diags[0, i] += - 2 / dx**2 * scale
            diags[1, i + 1] += 1 / dx**2 * scale
            diags[2, i - 1] += 1 / dx**2 * scale

        # Then fill in the y direction if it is not dirichlet line
        if not dc_bool:
            if 0 <= i <= nx - 1:
                if bcs['bottom'] == 'dirichlet':
                    dirichlet_line(diags, i, nx, ny)
                elif bcs['bottom'] == 'neumann':
                    neumann_line(diags, (i, (3, i + nx)), dy, scale)
            elif i >= nx * (ny - 1):
                if bcs['top'] == 'dirichlet':
                    dirichlet_line(diags, i, nx, ny)
                elif bcs['top'] == 'neumann':
                    neumann_line(diags, (i, (4, i - nx)), dy, scale)
            else:
                diags[0, i] += - 2 / dy**2 * scale
                diags[3, i + nx] += 1 / dy**2 * scale
                diags[4, i - nx] += 1 / dy**2 * scale

    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, nx, -nx]), shape=(nx * ny, nx * ny)))

def axisym_matrix(dx, dr, nx, nr, R, coeff, scale):
    """ Axisymmetric Screened Poisson equation matrix """
    diags = np.zeros((5, nx * nr))
    r = R.reshape(-1)

    # Filling the diagonals, first the down neumann bc,: the dirichlet bc and finally the interior nodes
    for i in range(nx * nr):
        if 0 < i < nx - 1:
            diags[0, i] = - (2 / dx**2 + 4 / dr**2 + coeff) * scale
            diags[1, i + 1] = 1 / dx**2 * scale
            diags[2, i - 1] = 1 / dx**2 * scale
            diags[3, i + nx] = 4 / dr**2 * scale
        elif i >= (nr - 1) * nx or i % nx == 0 or i % nx == nx - 1:
            diags[0, i] = 1
            diags[1, min(i + 1, nx * nr - 1)] = 0
            diags[2, max(i - 1, 0)] = 0
            diags[3, min(i + nx, nx * nr - 1)] = 0
            diags[4, max(i - nx, 0)] = 0
        else:
            diags[0, i] = - (2 / dx**2 + 2 / dr**2 + coeff) * scale
            diags[1, i + 1] = 1 / dx**2 * scale
            diags[2, i - 1] = 1 / dx**2 * scale
            diags[3, i + nx] = (1 + dr / (2 * r[i])) / dr**2 * scale
            diags[4, i - nx] = (1 - dr / (2 * r[i])) / dr**2 * scale

    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, nx, -nx]), shape=(nx * nr, nx * nr)))
