"""fedcomp: a small simulator for communication-compressed federated optimization."""

from .problems import Problem, LeastSquares, LogisticRegression, LinearEstimation
from .methods import Method, Message, GD, DCGD, DIANA, EF, EF21, ef21_stepsize
from .simulator import run, History
from .compressors import Compressor, Identity, RandK, TopK
from .bits import FLOAT_BITS, SEED_BITS, dense_bits, index_bits, randk_bits, topk_bits

__all__ = [
    "Problem", "LeastSquares", "LogisticRegression", "LinearEstimation",
    "Method", "Message", "GD", "DCGD", "DIANA", "EF", "EF21", "ef21_stepsize",
    "run", "History",
    "Compressor", "Identity", "RandK", "TopK",
    "FLOAT_BITS", "SEED_BITS", "dense_bits", "index_bits", "randk_bits", "topk_bits",
]
