########################################################################################################################
#                                                                                                                      #
#                                     Function to compute the first Fourier mode                                       #
#                                   as an initial guess for the Dirichlet Network                                      #
#                                                                                                                      #
#                                         Ekhi Ajuria, CERFACS, 29.04.2020                                             #
#                                                                                                                      #
########################################################################################################################

from itertools import repeat
from pathlib import Path

import pandas as pd
import yaml

def fourier_guess(BC, resX):
    ''' Will give a guess for the 2D network
    Inputs:
        - BC (array of size [bsz, 1, resY]): Potential on the BC for Dirichlet Network
        - resX (int): Size of the X direction. For the square datasets its trivial, but
                    might be handfull later on.
    Output:
        - guess (array of size [bsz, 1, resY, resX]): Potential guess
    '''
    assert BC.size(1) == 1, "Size for array not OK, dim should be 1, not {}".format(BC.size(1))
    bsz  = BC.size(0)
    resY = BC.size(2)

    guess.unsqueeze(3).expand((bsz, 1, resY, resX)) 
    
    return guess
