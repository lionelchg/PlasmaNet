########################################################################################################################
#                                                                                                                      #
#                                                  Dataset filtering                                                   #
#                                                                                                                      #
#                                      Guillaume BOGOPOLSKY, CERFACS, 19.05.2020                                       #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from numba import njit


def dataset_input_filter(dataset, input_cutoff_frequency, mesh_size, domain_length):
    """
    Filters in-place a dataset by cutting of frequencies in a Fourier transform.

    Parameters
    ----------
    dataset : Numpy array
            Dataset in Numpy format, of shape [dataset_size, Ny, Nz]

    input_cutoff_frequency : float
            Cutoff frequency in both directions

    mesh_size : int
            Size of the square mesh

    domain_length : float
            Physical length of the domain
    """
    for i in range(dataset.shape[0]):
        transf = np.fft.fft2(dataset[i])
        freq = np.fft.fftfreq(mesh_size, domain_length/mesh_size)
        cutoff_fourier_space(transf, freq, input_cutoff_frequency)
        dataset[i] = np.real(np.fft.ifft2(transf))


@njit
def cutoff_fourier_space(transf, freq, cutoff_frequency):
    """ Iterates over Fourier space and sets to zero all frequencies above cutoff_frequency. """
    size = transf.shape[0]
    for k in range(size):
        for j in range(size):
            if np.abs(freq[k]) > cutoff_frequency or np.abs(freq[j]) > cutoff_frequency:
                transf[k, j] = 0
