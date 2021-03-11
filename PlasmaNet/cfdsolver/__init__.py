"""
CfdSolver: Solving the plasma fluid equations

Lionel Cheng
CERFACS
"""

from .euler.plasma import PlasmaEuler
from .scalar.scalar import ScalarTransport
from .scalar.streamer import StreamerMorrow

__ALL__ = [
    PlasmaEuler,
    ScalarTransport,
    StreamerMorrow,
]