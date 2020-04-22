########################################################################################################################
#                                                                                                                      #
#                                                       Operators                                                      #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################
import numpy as np


def L1_error(y, y_th, ds, S):
  return np.sum(ds / S * abs(y - y_th))


def L2_error(y, y_th, ds, S):
  return np.sqrt(ds / S * np.sum((y - y_th)**2))


def Linf_error(y, y_th):
    return np.max(np.abs(y - y_th))


def errors(y, y_th, ds, S):
    return L1_error(y, y_th, ds, S), L2_error(y, y_th, ds, S), Linf_error(y, y_th)


def print_error(computed, analytical, ds, S, name):
    print('{} L1 error = {:.2e} - L2 error = {:.2e} - Linf error = {:.2e}'.format(name,
                                                                                  L1_error(computed, analytical, ds, S),
                                                                                  L2_error(computed, analytical, ds, S),
                                                                                  Linf_error(computed, analytical)))
def derivative(y, x, dx):
    dy = np.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * dx)
    dy[0] = 4 * y[1] - 3 * y[0] - y[2]
    dy[-1] = - (4 * y[-2] - 3 * y[-1] - y[-3])
    return dy

def div(field, dx, dy, nx, ny, order=2):

    divergence = np.zeros((ny, nx))

    if order == 2:
        divergence[1:-1, 1:-1] = (field[0, 1:-1, 2:] - field[0, 1:-1, :-2]) / (2 * dx) + \
                                 (field[1, 2:, 1:-1] - field[1, :-2, 1:-1]) / (2 * dy)
    elif order == 4:
        divergence[1:-1, 1:-1] = (field[0, 1:-1, 2:] - field[0, 1:-1, :-2]) / (2 * dx) + \
                                       (field[1, 2:, 1:-1] - field[1, :-2, 1:-1]) / (2 * dy)
        divergence[2:-2, 2:-2] = (- field[0, 2:-2, 4:] + 8 * field[0, 2:-2, 3:-1]
                                  - 8 * field[0, 2:-2, 1:-3] + field[0, 2:-2, :-4]) / (12 * dx) + \
                                 (- field[1, 4:, 2:-2] + 8 * field[1, 3:-1, 2:-2]
                                  - 8 * field[1, 1:-3, 2:-2] + field[1, :-4, 2:-2]) / (12 * dy)

    # array sides except corners (respectively upper, lower, left and right sides)
    divergence[0, 1:-1] = (field[0, 0, 2:] - field[0, 0, :-2]) / (2 * dx) + \
                          (4 * field[1, 1, 1:-1] - 3 * field[1, 0, 1:-1] - field[1, 2, 1:-1]) / (2 * dy)
    divergence[-1, 1:-1] = (field[0, -1, 2:] - field[0, -1, :-2]) / (2 * dx) + \
                           (3 * field[1, -1, 1:-1] - 4 * field[1, -2, 1:-1] + field[1, -3, 1:-1]) / (2 * dy)
    divergence[1:-1, 0] = (4 * field[0, 1:-1, 1] - 3 * field[0, 1:-1, 0] - field[0, 1:-1, 2]) / \
                          (2 * dx) + (field[1, 2:, 0] - field[1, :-2, 0]) / (2 * dy)
    divergence[1:-1, -1] = (3 * field[0, 1:-1, -1] - 4 * field[0, 1:-1, -2] + field[0, 1:-1, -3]) / \
                           (2 * dx) + (field[1, 2:, -1] - field[1, :-2, -1]) / (2 * dy)

    # corners (respectively upper left, upper right, lower left and lower right)
    divergence[0, 0] = (4 * field[0, 0, 1] - 3 * field[0, 0, 0] - field[0, 0, 2]) / (2 * dx) + \
                       (4 * field[0, 1, 0] - 3 * field[0, 0, 0] - field[0, 2, 0]) / (2 * dy)
    divergence[-1, 0] = (4 * field[0, -1, 1] - 3 * field[0, -1, 0] - field[0, -1, 2]) / (2 * dx) + \
                        (3 * field[1, -1, 0] - 4 * field[1, -2, 0] + field[1, -3, 0]) / (2 * dy)
    divergence[0, -1] = (3 * field[0, 0, -1] - 4 * field[0, 0, -2] + field[0, 0, -3]) / (2 * dx) +\
                        (4 * field[1, 1, -1] - 3 * field[1, 0, -1] - field[1, 2, -1]) / (2 * dy)
    divergence[-1, -1] = (3 * field[0, -1, -1] - 4 * field[0, -1, -2] + field[0, -1, -3]) / (2 * dx) + \
                         (3 * field[1, -1, -1] - 4 * field[1, -2, -1] + field[1, -3, -1]) / (2 * dy)
    return divergence


def lapl(field, dx, dy, nx, ny, order=2, b=0):

    laplacian = np.zeros((ny, nx))

    laplacian[1:-1, 1:-1] = (field[2:, 1:-1] + field[:-2, 1:-1] - 2 * field[1:-1, 1:-1]) / dy**2 + \
                            (field[1:-1, 2:] + field[1:-1, :-2] - 2 * field[1:-1, 1:-1]) / dx**2
    if order == 2:
        laplacian[1:-1, 1:-1] = (1 - b) * ((field[2:, 1:-1] + field[:-2, 1:-1] - 2 * field[1:-1, 1:-1]) / dy**2 +
                            (field[1:-1, 2:] + field[1:-1, :-2] - 2 * field[1:-1, 1:-1]) / dx**2) + \
                            b * (field[2:, 2:] + field[2:, :-2] + field[:-2, :-2] + field[:-2, 2:] - 4 * field[1:-1, 1:-1]) \
                            / (2 * dx**2)
    elif order == 4:
        laplacian[2:-2, 2:-2] = (- field[4:, 2:-2] + 16 * field[3:-1, 2:-2] - 30 * field[2:-2, 2:-2] + 16 * field[1:-3, 2:-2]
                                 - field[:-4, 2:-2]) / (12 * dy**2) + \
                                (- field[2:-2, 4:] + 16 * field[2:-2, 3:-1] - 30 * field[2:-2, 2:-2] + 16 * field[2:-2, 1:-3]
                                 - field[2:-2, :-4]) / (12 * dx**2)
        laplacian[1, 1:-1] = ((field[2, 1:-1] + field[0, 1:-1] - 2 * field[1, 1:-1]) / dy**2 +
                             (field[1, 2:] + field[1, :-2] - 2 * field[1, 1:-1]) / dx**2)
        laplacian[-2, 1:-1] = ((field[-1, 1:-1] + field[-3, 1:-1] - 2 * field[-2, 1:-1]) / dy**2 +
                              (field[-2, 2:] + field[-2, :-2] - 2 * field[-2, 1:-1]) / dx**2)
        laplacian[1:-1, 1] = (field[2:, 1] + field[:-2, 1] - 2 * field[1:-1, 1]) / dy**2 + \
                             (field[1:-1, 2] + field[1:-1, 0] - 2 * field[1:-1, 1]) / dx**2
        laplacian[1:-1, -2] = (field[2:, -2] + field[:-2, -2] - 2 * field[1:-1, -2]) / dy**2 + \
                              (field[1:-1, -1] + field[1:-1, -3] - 2 * field[1:-1, -2]) / dx**2

    laplacian[0, 1:-1] = \
        (2 * field[0, 1:-1] - 5 * field[1, 1:-1] + 4 * field[2, 1:-1] - field[3, 1:-1]) / dy**2 + \
        (field[0, 2:] + field[0, :-2] - 2 * field[0, 1:-1]) / dx**2
    laplacian[-1, 1:-1] = \
        (2 * field[-1, 1:-1] - 5 * field[-2, 1:-1] + 4 * field[-3, 1:-1] - field[-4, 1:-1]) / dy**2 + \
        (field[-1, 2:] + field[-1, :-2] - 2 * field[-1, 1:-1]) / dx**2
    laplacian[1:-1, 0] = \
        (field[2:, 0] + field[:-2, 0] - 2 * field[1:-1, 0]) / dy**2 + \
        (2 * field[1:-1, 0] - 5 * field[1:-1, 1] + 4 * field[1:-1, 2] - field[1:-1, 3]) / dx**2
    laplacian[1:-1, -1] = \
        (field[2:, -1] + field[:-2, -1] - 2 * field[1:-1, -1]) / dy**2 + \
        (2 * field[1:-1, -1] - 5 * field[1:-1, -2] + 4 * field[1:-1, -3] - field[1:-1, -4]) / dx**2

    # corners (respectively upper left, upper right, lower left and lower right)
    laplacian[0, 0] = \
        (2 * field[0, 0] - 5 * field[1, 0] + 4 * field[2, 0] - field[3, 0]) / dy**2 + \
        (2 * field[0, 0] - 5 * field[0, 1] + 4 * field[0, 2] - field[0, 3]) / dx**2
    laplacian[0, -1] = \
        (2 * field[0, -1] - 5 * field[1, -1] + 4 * field[2, -1] - field[3, -1]) / dy**2 + \
        (2 * field[0, -1] - 5 * field[0, -2] + 4 * field[0, -3] - field[0, -4]) / dx**2
    laplacian[-1, 0] = \
        (2 * field[-1, 0] - 5 * field[-2, 0] + 4 * field[-3, 0] - field[-4, 0]) / dy**2 + \
        (2 * field[-1, 0] - 5 * field[-1, 1] + 4 * field[-1, 2] - field[-1, 3]) / dx**2
    laplacian[-1, -1] = \
        (2 * field[-1, -1] - 5 * field[-2, -1] + 4 * field[-3, -1] - field[-4, -1]) / dy**2 + \
        (2 * field[0, -1] - 5 * field[0, -2] + 4 * field[0, -3] - field[0, -4]) / dx**2

    return laplacian


def grad(field, dx, dy, nx, ny):
    gradient = np.zeros((2, ny, nx))

    gradient[0, :, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx)
    gradient[0, :, 0] = (4 * field[:, 1] - 3 * field[:, 0] - field[:, 2]) / (2 * dx)
    gradient[0, :, -1] = - (4 * field[:, -2] - 3 * field[:, -1] - field[:, -3]) / (2 * dx)

    gradient[1, 1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dy)
    gradient[1, 0, :] = (4 * field[1, :] - 3 * field[0, :] - field[2, :]) / (2 * dy)
    gradient[1, -1, :] = - (4 * field[-2, :] - 3 * field[-1, :] - field[-3, :]) / (2 * dy)

    return gradient

def scalar_rot(field, dx, dy, nx, ny):
    rotational = np.zeros((ny, nx))

    # first compute dfield_y / dx
    rotational[:, 1:-1] = (field[1, :, 2:] - field[1, :, :-2]) / (2 * dx)
    rotational[:, 0] = (4 * field[1, :, 1] - 3 * field[1, :, 0] - field[1, :, 2]) / (2 * dx)
    rotational[:, -1] = (3 * field[1, :, -1] - 4 * field[1, :, -2] + field[1, :, -3]) / (2 * dx)

    # second compute dfield_x / dy
    rotational[1:-1, :] -= (field[0, 2:, :] - field[0, :-2, :]) / (2 * dy)
    rotational[0, :] -= (4 * field[0, 1, :] - 3 * field[0, 0, :] - field[0, 2, :]) / (2 * dy)
    rotational[-1, :] -= (3 * field[0, -1, :] - 4 * field[0, -2, :] + field[0, -3, :]) / (2 * dy)

    return rotational
