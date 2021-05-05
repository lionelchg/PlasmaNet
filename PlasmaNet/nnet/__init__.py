"""
PlasmaNet: Solving the electrostatic Poisson equation for plasma simulations

Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria
CERFACS
"""

__author__ = 'Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria'
__mail__ = ''
__version__ = '1.0'

from .data import data_loaders
from .model import loss, metric
from .trainer.trainer import Trainer
from .parse_config import ConfigParser

__ALL__ = [
    data_loaders,
    loss,
    metric,
    Trainer,
    ConfigParser,
]
