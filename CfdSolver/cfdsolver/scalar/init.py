import numpy as np

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    """ Gaussian function """
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2
                              - ((y - y0) / sigma_y) ** 2)

def triangle(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    """ Triangle function """
    return (amplitude * np.maximum(1 - abs((x - x0) / sigma_x), np.zeros_like(x)) 
                * np.maximum(1 - abs((y - y0) / sigma_y), np.zeros_like(x)))

func_dict = {'gaussian': gaussian, 'triangle': triangle}
