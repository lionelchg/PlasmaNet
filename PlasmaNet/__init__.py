"""
PlasmaNet: Solving the electrostatic Poisson equation for plasma simulations

Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria
CERFACS
"""

__author__ = 'Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria'
__mail__ = ''
__version__ = '0.1'

from .Models.multiscalenet import MultiSimpleNet
from .misc import *
from .Operators import *

__ALL__ = [
    MultiSimpleNet
]
