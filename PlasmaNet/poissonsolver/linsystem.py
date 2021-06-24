########################################################################################################################
#                                                                                                                      #
#                                        Routines concerning the linear system                                         #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from scipy import sparse


def cartesian_matrix(dx: float, dy: float, nx: int, ny: int, scale: float, bcs: dict) -> sparse.csc_matrix:
    """ Generate a cartesian matrix with prescribed boundary conditions
    (Dirichlet or Neumann mixed)

    :param dx: x resolution
    :type dx: float
    :param dy: y resolution
    :type dy: float
    :param nx: number of nodes in x direction
    :type nx: int
    :param ny: number of nodes in y direction
    :type ny: int
    :param scale: scaling factor for the matrix coefficients
    :type scale: float
    :param bcs: boundary conditions (either string or list of strings)
    :type bcs: dict
    :return: matrix of the Poisson equation
    :rtype: sparse.csc_matrix
    """
    diags = np.zeros((5, nx * ny))

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


def neumann_line(diags: np.array, diag_indices: tuple, res: float, scale: float) -> None:
    """ Fill a Neumann boundary condition line in the Poisson matrix

    :param diags: the diagonals used to fill the matrix
    :type diags: np.array
    :param diag_indices: the indices of the diagonals to change
    :type diag_indices: tuple
    :param res: the resolution
    :type res: float
    :param scale: scaling factor
    :type scale: float
    """
    diags[0, diag_indices[0]] += - 2 / res**2 * scale
    diags[diag_indices[1]] += 2 / res**2 * scale


def dirichlet_line(diags: np.array, i: int, nx: int, ny: int) -> None:
    """ Fill a Dirichlet boundary condition line in the Poisson matrix

    :param diags: the diagonals used to fill the matrix
    :type diags: np.array
    :param i: line index
    :type i: int
    :param nx: number of nodes in x direction
    :type nx: int
    :param ny: number of nodes in y direction
    :type ny: int
    """
    diags[0, i] = 1
    diags[1, min(i + 1, nx * ny - 1)] = 0
    diags[2, max(i - 1, 0)] = 0
    diags[3, min(i + nx, nx * ny - 1)] = 0
    diags[4, max(i - nx, 0)] = 0


def matrix_axisym(dx, dr, nx, nr, R, scale):
    """ Build the matrix for the axisymmetric configuration. """
    diags = np.zeros((5, nx * nr))

    r = R.reshape(-1)

    # Filling the diagonals, first the down neumann bc, then the dirichlet bc and finally the interior nodes
    for i in range(nx * nr):
        if 0 < i < nx - 1:
            diags[0, i] = - (2 / dx**2 + 4 / dr**2) * scale
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
            diags[0, i] = - (2 / dx**2 + 2 / dr**2) * scale
            diags[1, i + 1] = 1 / dx**2 * scale
            diags[2, i - 1] = 1 / dx**2 * scale
            diags[3, i + nx] = (1 + dr / (2 * r[i])) / dr**2 * scale
            diags[4, i - nx] = (1 - dr / (2 * r[i])) / dr**2 * scale

    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, nx, -nx]), shape=(nx * nr, nx * nr)))


def matrix_cart_perio(dx, dy, nx, ny, scale):
    """ Creation of the matrix for the full periodic problem in 
    Cartesian geometry """
    diags = np.zeros((9, nx * ny))

    # Filling the diagonals, first in x and then in y (down-up, left-right)
    for i in range(nx * ny): 
        # Start by filling in x direction
        if i % nx == 0:
            diags[0, i] += - 2 / dx**2 * scale
            diags[1, i + 1] += 1 / dx**2 * scale
            diags[5, i + nx - 1] += 1 / dx**2 * scale
        elif i % nx == nx - 1:
            diags[0, i] += - 2 / dx**2 * scale
            diags[2, i - 1] += 1 / dx**2 * scale
            diags[6, i - (nx - 1)] += 1 / dx**2 * scale
        else:
            diags[0, i] += - 2 / dx**2 * scale
            diags[1, i + 1] += 1 / dx**2 * scale
            diags[2, i - 1] += 1 / dx**2 * scale

        # Then y direction
        if i // nx == 0:
            diags[0, i] += - 2 / dy**2 * scale
            diags[3, i + nx] += 1 / dy**2 * scale
            diags[7, i + nx * (ny - 1)] += 1 / dy**2 * scale
        elif i // nx == ny - 1:
            diags[0, i] += - 2 / dy**2 * scale 
            diags[4, i - nx] += 1 / dy**2 * scale
            diags[8, i - nx * (ny - 1)] += 1 / dx**2 * scale
        else:
            diags[0, i] += - 2 / dy**2 * scale
            diags[3, i + nx] += 1 / dy**2 * scale
            diags[4, i - nx] += 1 / dy**2 * scale

    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, nx, -nx, nx - 1, -(nx - 1), nx * (ny - 1), -nx * (ny - 1)]), 
                          shape=(nx * ny, nx * ny)))


def matrix_cart_perio_x(dx, dy, nx, ny, scale):
    """ Creation of the matrix for x-periodic / y-dirichlet problem in 
    Cartesian geometry """
    diags = np.zeros((7, nx * ny))

    # Filling the diagonals, first the down neumann bc, then the dirichlet bc and finally the interior nodes
    for i in range(nx * ny): 
        # Start by filling in x direction
        if i % nx == 0:
            diags[0, i] += - 2 / dx**2 * scale
            diags[1, i + 1] += 1 / dx**2 * scale
            diags[5, i + nx - 1] += 1 / dx**2 * scale
        elif i % nx == nx - 1:
            diags[0, i] += - 2 / dx**2 * scale
            diags[2, i - 1] += 1 / dx**2 * scale
            diags[6, i - (nx - 1)] += 1 / dx**2 * scale
        else:
            diags[0, i] += - 2 / dx**2 * scale
            diags[1, i + 1] += 1 / dx**2 * scale
            diags[2, i - 1] += 1 / dx**2 * scale

        # Then y direction
        if 0 <= i <= nx - 1 or i >= nx * (ny - 1):
            diags[0, i] = 1
            diags[1, min(i + 1, nx * ny - 1)] = 0
            diags[2, max(i - 1, 0)] = 0
            diags[3, min(i + nx, nx * ny - 1)] = 0
            diags[4, max(i - nx, 0)] = 0
            diags[5, min(i + nx - 1, nx * ny - 1)] = 0
            diags[6, max(i - (nx - 1), 0)] = 0
        else:
            diags[0, i] += - 2 / dy**2 * scale
            diags[3, i + nx] += 1 / dy**2 * scale
            diags[4, i - nx] += 1 / dy**2 * scale
        
    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, nx, -nx, nx - 1, -(nx - 1)]), shape=(nx * ny, nx * ny)))


def impose_dirichlet(rhs: np.ndarray, bcs: dict) -> None:
    """ Impose Dirichlet boundary conditions to the rhs vector

    :param rhs: rhs vector of the Poisson equation
    :type rhs: 2D-np.ndarray
    :param bcs: dictionnary of boundary conditions
    :type bcs: dict
    """
    if 'left' in bcs:
        rhs[:, 0] = bcs['left']
    
    if 'right' in bcs: 
        rhs[:, -1] = bcs['right']
    
    if 'bottom' in bcs:
        rhs[0, :] = bcs['bottom']

    if 'top' in bcs:
        rhs[-1, :] = bcs['top']
