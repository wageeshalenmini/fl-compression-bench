"""Bit accounting for uplink messages.

Every method must report how many bits each device uploads in each round.
All formulas live here so that the accounting is the same for every method.
Compressor-specific formulas (Rand-k, Top-k, ...) are added in step 3.
"""

FLOAT_BITS = 32  # one transmitted real number, as a 32-bit float


def dense_bits(d: int) -> int:
    """Bits needed to send a full vector of d real numbers."""
    return FLOAT_BITS * d
