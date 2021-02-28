import numpy as np
from scipy import interpolate

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    """ Gaussian function """
    return amplitude * np.exp(-((x - x0) / sigma_x)**2
                              - ((y - y0) / sigma_y)**2)

def two_gaussians(x, y, amplitude, x0, y0, sigma_x, sigma_y, x01, y01, sigma_x1, sigma_y1):
    """ Gaussian function """
    return amplitude * (np.exp(-((x - x0) / sigma_x)**2 - ((y - y0) / sigma_y)**2) + 
                        np.exp(-((x - x01) / sigma_x1)**2 - ((y - y01) / sigma_y1)**2))

def triangle(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    """ Triangle function """
    return (amplitude * np.maximum(1 - abs((x - x0) / sigma_x), np.zeros_like(x)) 
                * np.maximum(1 - abs((y - y0) / sigma_y), np.zeros_like(x)))

def step(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    """ Step function """
    return (amplitude * np.where(abs(x - x0) / sigma_x < 0.5, np.ones_like(x), np.zeros_like(x))
            * np.where(abs(y - y0) / sigma_y < 0.5, np.ones_like(x), np.zeros_like(x)))

def sin2D(x, y, amplitude, Lx, Ly, n, m):
    """ 2D sines mode """
    return amplitude * np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly)

def gaussians(x, y, params):
    """ Multiple gaussians with multiple amplitude """
    profile = np.zeros_like(x)
    ngauss = int(len(params) / 5)
    params = np.array(params).reshape(ngauss, 5)
    for index in range(ngauss):
        profile += gaussian(x, y, *params[index, :])
    return profile

def random(X, Y, amplitude, n_res, sigma):
    xmin, xmax, npts = np.min(X), np.max(X), len(X[0, :])
    n_lower = int(npts / n_res)
    x_lower, y_lower = np.linspace(xmin, xmax, n_lower), np.linspace(xmin, xmax, n_lower)
    z_lower = amplitude * (2 * np.random.random((n_lower, n_lower)) - 1)
    f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')
    x, y = np.linspace(xmin, xmax, npts), np.linspace(xmin, xmax, npts)
    x0, y0 = (xmax - xmin) / 2, (xmax - xmin) / 2
    return f(x, y) * np.exp(-((X - x0)**2 / sigma**2 + (Y - y0)**2 / sigma**2))