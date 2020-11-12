"""
CfdSolver: Solving the plasma fluid equations

Lionel Cheng
CERFACS
"""

__author__ = 'Lionel Cheng'
__mail__ = ''
__version__ = '0.5'

from .euler.euler import Euler
from .euler.plasma import PlasmaEuler
from .scalar.scalar import ScalarTransport
from .scalar.streamer import StreamerMorrow

__ALL__ = [
    Euler,
    ScalarTransport,
    StreamerMorrow,
    PlasmaEuler
]
