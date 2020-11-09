import numpy as np

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)
