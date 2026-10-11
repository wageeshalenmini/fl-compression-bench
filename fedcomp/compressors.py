"""Compression operators.

A compressor maps a vector x to a cheaper-to-send vector C(x). Calling one
returns the vector the server would reconstruct, and the number of bits the
message costs.

Two classes are used in the theory:

  unbiased     E[C(x)] = x   and   E||C(x) - x||^2 <= omega * ||x||^2
  contractive  E||C(x) - x||^2 <= (1 - alpha) * ||x||^2,   0 < alpha <= 1

Each compressor reports the constants it satisfies through `omega` and
`alpha` (None when it is not in that class). `setup(d)` must be called
before these are read, because they depend on the dimension.

Randomness always comes from the generator passed in, never from global
state. Each device owns its own generator, so masks are independent across
devices, which is what the variance formulas assume.
"""

from __future__ import annotations

import numpy as np

from .bits import dense_bits, randk_bits, topk_bits


class Compressor:
    name = "base"

    def setup(self, d: int) -> None:
        self.d = d

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    @property
    def unbiased(self) -> bool:
        return self.omega is not None

    omega: float | None = None  # variance parameter, if unbiased
    alpha: float | None = None  # contraction parameter, if contractive


class Identity(Compressor):
    """No compression. Sends the full vector."""

    name = "Identity"
    omega = 0.0
    alpha = 1.0

    def __call__(self, x, rng=None):
        return x.copy(), dense_bits(len(x))


class RandK(Compressor):
    """Keep k coordinates chosen uniformly at random.

    scaled=True  (default): multiply the kept values by d/k. Unbiased, with
                 E||C(x) - x||^2 = (d/k - 1) ||x||^2 exactly.
    scaled=False: keep the values as they are. Biased but contractive, with
                 E||C(x) - x||^2 = (1 - k/d) ||x||^2 exactly.
    """

    def __init__(self, k: int, scaled: bool = True):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = int(k)
        self.scaled = scaled
        self.name = f"Rand-{self.k}" + ("" if scaled else " (unscaled)")

    def setup(self, d):
        if self.k > d:
            raise ValueError(f"k = {self.k} is larger than the dimension d = {d}")
        self.d = d
        if self.scaled:
            self.omega = d / self.k - 1.0
            self.alpha = 1.0 if self.k == d else None
        else:
            self.omega = 0.0 if self.k == d else None
            self.alpha = self.k / d

    def __call__(self, x, rng):
        d = len(x)
        idx = rng.choice(d, size=self.k, replace=False)
        out = np.zeros(d)
        out[idx] = x[idx] * (d / self.k if self.scaled else 1.0)
        return out, randk_bits(self.k)


class TopK(Compressor):
    """Keep the k coordinates of largest magnitude, unscaled.

    Deterministic and biased. Contractive with alpha = k/d:
    ||C(x) - x||^2 <= (1 - k/d) ||x||^2, with equality when all entries
    have the same magnitude.
    """

    def __init__(self, k: int):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = int(k)
        self.name = f"Top-{self.k}"

    def setup(self, d):
        if self.k > d:
            raise ValueError(f"k = {self.k} is larger than the dimension d = {d}")
        self.d = d
        self.alpha = self.k / d
        self.omega = 0.0 if self.k == d else None

    def __call__(self, x, rng=None):
        d = len(x)
        out = np.zeros(d)
        if self.k == d:
            out[:] = x
        else:
            idx = np.argpartition(np.abs(x), d - self.k)[d - self.k:]
            out[idx] = x[idx]
        return out, topk_bits(self.k, d)
