"""
GaussNet: Solving the electrostatic Poisson equation for plasma simulations

Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria
CERFACS
"""

__author__ = 'Guillaume Bogopolsky, Lionel Cheng, Akhi Ajuria'
__mail__ = ''
__version__ = '0.1'

from .Models.fluidnet import MultiSimpleNet

__ALL__ = [
    MultiSimpleNet
]
