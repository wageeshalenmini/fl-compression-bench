"""fedcomp: a small simulator for communication-compressed federated optimization."""

from .problems import Problem, LeastSquares, LogisticRegression
from .methods import Method, Message, GD
from .simulator import run, History
from .bits import FLOAT_BITS, dense_bits

__all__ = [
    "Problem", "LeastSquares", "LogisticRegression",
    "Method", "Message", "GD",
    "run", "History",
    "FLOAT_BITS", "dense_bits",
]
