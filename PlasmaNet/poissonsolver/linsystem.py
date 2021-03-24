########################################################################################################################
#                                                                                                                      #
#                                        Routines concerning the linear system                                         #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from scipy import sparse


def laplace_square_matrix(n_points):
    diags = np.zeros((5, n_points * n_points))

    # Filling the diagonal
    diags[0, :] = - 4 * np.ones(n_points ** 2)
    diags[1, :] = np.ones(n_points ** 2)
    diags[2, :] = np.ones(n_points ** 2)
    diags[3, :] = np.ones(n_points ** 2)
    diags[4, :] = np.ones(n_points ** 2)

    # Correcting the diagonal to take into account dirichlet boundary conditions, first diagonal to one and correct 
    # all the other ones
    for i in range(n_points ** 2):
        if i < n_points or i >= (n_points - 1) * n_points or i % n_points == 0 or i % n_points == n_points - 1:
            diags[0, i] = 1
            diags[1, min(i + 1, n_points ** 2 - 1)] = 0
            diags[2, max(i - 1, 0)] = 0
            diags[3, min(i + n_points, n_points ** 2 - 1)] = 0
            diags[4, max(i - n_points, 0)] = 0

    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, n_points, -n_points]), shape=(n_points ** 2, n_points ** 2)))


def dirichlet_bc(rhs, n_points, up, down, left, right):
    # filling of the four boundaries, inversion of up and down ?
    rhs[:n_points] = up
    rhs[n_points * (n_points - 1):] = down
    rhs[:n_points * (n_points - 1) + 1:n_points] = left
    rhs[n_points - 1::n_points] = right
    # mean approximation in case the potential is not continuous across boundaries
    rhs[0] = 0.5 * (up[0] + left[0])
    rhs[n_points - 1] = 0.5 * (up[-1] + right[1])
    rhs[-n_points] = 0.5 * (left[-1] + down[0])
    rhs[-1] = 0.5 * (right[-1] + down[-1])


def matrix_cart(dx, dy, nx, ny, scale, down_bc='dirichlet'):
    """ Creation of the matrix for the down neumann, left/up/right dirichlet
    or full dirichlet bc Poisson problem in cartesian geometry """
    diags = np.zeros((5, nx * ny))

    if down_bc == 'neumann':
        # Filling the diagonals, first the down neumann bc, then the dirichlet bc and finally the interior nodes
        for i in range(nx * ny):
            if 0 < i < nx - 1:
                diags[0, i] = - (2 / dx**2 + 2 / dy**2) * scale
                diags[1, i + 1] = 1 / dx**2 * scale
                diags[2, i - 1] = 1 / dx**2 * scale
                diags[3, i + nx] = 2 / dy**2 * scale
            elif i >= (ny - 1) * nx or i % nx == 0 or i % nx == nx - 1:
                diags[0, i] = 1
                diags[1, min(i + 1, nx * ny - 1)] = 0
                diags[2, max(i - 1, 0)] = 0
                diags[3, min(i + nx, nx * ny - 1)] = 0
                diags[4, max(i - nx, 0)] = 0
            else:
                diags[0, i] = - (2 / dx**2 + 2 / dy**2) * scale
                diags[1, i + 1] = 1 / dx**2 * scale
                diags[2, i - 1] = 1 / dx**2 * scale
                diags[3, i + nx] = 1 / dy**2 * scale
                diags[4, i - nx] = 1 / dy**2 * scale
    elif down_bc == 'dirichlet':
        # Filling the diagonals, first the down neumann bc, then the dirichlet bc and finally the interior nodes
        for i in range(nx * ny):
            if 0 < i < (nx - 1) or i >= (ny - 1) * nx or i % nx == 0 or i % nx == nx - 1:
                diags[0, i] = 1
                diags[1, min(i + 1, nx * ny - 1)] = 0
                diags[2, max(i - 1, 0)] = 0
                diags[3, min(i + nx, nx * ny - 1)] = 0
                diags[4, max(i - nx, 0)] = 0
            else:
                diags[0, i] = - (2 / dx**2 + 2 / dy**2) * scale
                diags[1, i + 1] = 1 / dx**2 * scale
                diags[2, i - 1] = 1 / dx**2 * scale
                diags[3, i + nx] = 1 / dy**2 * scale
                diags[4, i - nx] = 1 / dy**2 * scale

    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, nx, -nx]), shape=(nx * ny, nx * ny)))


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


def dirichlet_bc_axi(rhs, nx, nr, up, left, right):
    """ Apply the three dirichlet boundary conditions for axisymmetric test case. """
    rhs[nx * (nr - 1):] = up
    rhs[:nx * (nr - 1) + 1:nx] = left
    rhs[nx - 1::nx] = right
    # mean approximation in case the potential is not continuous across boundaries
    rhs[-nx] = 0.5 * (left[-1] + up[0])
    rhs[-1] = 0.5 * (right[-1] + up[-1])


def impose_dc_bc(rhs, nx, ny, args):
    """ Apply boundary conditions on the boundaries
    If there is only one boundary with one value then it is to impose the potential at one point
    If there are 3 boundaries then it is ordered up, left, right
    If there are 4 boundaries then down, up, left, right """
    if len(args) == 1:
        # rhs[0] = 0
        # rhs[:nx] = 0.0
        # rhs[nx * (ny - 1):] = 0.0
        pass
    elif len(args) == 3:
        up, left, right = args
        # filling of the three dirichlet boundaries for axisymmetric test case
        rhs[nx * (ny - 1):] = up
        rhs[:nx * (ny - 1) + 1:nx] = left
        rhs[nx - 1::nx] = right
        # mean approximation in case the potential is not continuous across boundaries
        rhs[-nx] = 0.5 * (left[-1] + up[0])
        rhs[-1] = 0.5 * (right[-1] + up[-1])
    elif len(args) == 4:
        down, up, left, right = args
        # filling of the three dirichlet boundaries for axisymmetric test case
        rhs[:nx] = down
        rhs[nx * (ny - 1):] = up
        rhs[:nx * (ny - 1) + 1:nx] = left
        rhs[nx - 1::nx] = right
        # mean approximation in case the potential is not continuous across boundaries
        rhs[-nx] = 0.5 * (left[-1] + up[0])
        rhs[-1] = 0.5 * (right[-1] + up[-1])
        rhs[0] = 0.5 * (left[0] + down[0])
        rhs[nx - 1] = 0.5 * (right[0] + down[-1])

def matrix_cart_perio(dx, dy, nx, ny, scale):
    """ Creation of the matrix for the full neumann problem in 
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

def cartesian_matrix(dx:float, dy:float, nx:int, ny:int, scale:float, bcs:dict) -> sparse.csc_matrix:
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



def neumann_line(diags:np.array, diag_indices:tuple, res:float, scale:float) -> None:
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

def dirichlet_line(diags:np.array, i:int, nx:int, ny:int) -> None:
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

def impose_dirichlet(rhs:np.ndarray, nx:int, ny:int, bcs:dict) -> None:
    """ Impose Dirichlet boundary conditions to the rhs vector

    :param rhs: rhs vector of the Poisson equation
    :type rhs: np.ndarray
    :param nx: number of nodes in x direction
    :type nx: int
    :param ny: number of nodes in y direction
    :type ny: int
    :param bcs: dictionnary of boundary conditions
    :type bcs: dict
    """
    if 'left' in bcs:
        rhs[:nx * (ny - 1) + 1:nx] = bcs['left']
    
    if 'right' in bcs: 
        rhs[nx - 1::nx] = bcs['right']
    
    if 'bottom' in bcs:
        rhs[:nx] = bcs['bottom']

    if 'top' in bcs:
        rhs[nx * (ny - 1):] = bcs['top']