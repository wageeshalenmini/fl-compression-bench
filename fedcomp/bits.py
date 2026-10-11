"""Bit accounting for uplink messages.

Every method must report how many bits each device uploads in each round.
All formulas live here so that the accounting is the same for every method.
"""

import math

FLOAT_BITS = 32  # one transmitted real number, as a 32-bit float
SEED_BITS = 32   # one random seed, from which the server regenerates a mask


def dense_bits(d: int) -> int:
    """Bits needed to send a full vector of d real numbers."""
    return FLOAT_BITS * d


def index_bits(d: int) -> int:
    """Bits needed to name one position in a vector of length d."""
    return max(1, math.ceil(math.log2(d)))


def randk_bits(k: int) -> int:
    """Rand-k: k values plus the seed of the mask.

    The device and the server generate the same random mask from a shared
    seed (as in Konecny et al., 2016), so no positions are transmitted.
    """
    return FLOAT_BITS * k + SEED_BITS


def topk_bits(k: int, d: int) -> int:
    """Top-k: k values, and for each one its position.

    The positions depend on the data, so they cannot come from a seed.
    """
    return k * (FLOAT_BITS + index_bits(d))
