"""fedcomp: a small simulator for communication-compressed federated optimization."""

from .problems import Problem, LeastSquares, LogisticRegression
from .methods import Method, Message, GD
from .simulator import run, History
from .compressors import Compressor, Identity, RandK, TopK
from .bits import FLOAT_BITS, SEED_BITS, dense_bits, index_bits, randk_bits, topk_bits

__all__ = [
    "Problem", "LeastSquares", "LogisticRegression",
    "Method", "Message", "GD",
    "run", "History",
    "Compressor", "Identity", "RandK", "TopK",
    "FLOAT_BITS", "SEED_BITS", "dense_bits", "index_bits", "randk_bits", "topk_bits",
]
