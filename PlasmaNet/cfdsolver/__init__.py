"""
CfdSolver: Solving the plasma fluid equations

Lionel Cheng, Ekhi Ajuria, Guillaume Bogopolsky
CERFACS
"""

from .euler.plasma import PlasmaEuler
from .euler.plasma_dl import PlasmaEulerDL
from .scalar.scalar import ScalarTransport
from .scalar.streamer import StreamerMorrow

__ALL__ = [
    PlasmaEuler,
    PlasmaEulerDL,
    ScalarTransport,
    StreamerMorrow,
]
