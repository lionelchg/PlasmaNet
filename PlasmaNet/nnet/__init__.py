"""
PlasmaNet: Solving the electrostatic Poisson equation for plasma simulations

Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria
CERFACS
"""

__author__ = 'Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria'
__mail__ = ''
__version__ = '0.1'

from .data import data_loaders
from .model import loss, metric, multiscalenet, dirichletnet, unet
from .trainer.trainer import Trainer
from .parse_config import ConfigParser

__ALL__ = [
    data_loaders,
    loss,
    metric,
    multiscalenet,
    dirichletnet,
    unet,
    Trainer,
    ConfigParser,
]
