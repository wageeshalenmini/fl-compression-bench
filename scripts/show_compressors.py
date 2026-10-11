"""Step 3 demo: apply each compressor to the example vector from the notes.

    python scripts/show_compressors.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcomp import Identity, RandK, TopK  # noqa: E402

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

g = np.array([0.8, -0.3, 0.05, 1.2, -0.6])
rng = np.random.default_rng(0)
draws = 100_000

print(f"g = {g}    ||g||^2 = {g @ g:.4f}\n")
print(f"{'compressor':<20} {'one output':<38} {'bits':>5} {'mean sq. error':>15} {'theory':>8}")
for comp in [Identity(), RandK(2), RandK(2, scaled=False), TopK(2)]:
    comp.setup(len(g))
    out, bits = comp(g, rng)
    err = np.mean([np.sum((comp(g, rng)[0] - g) ** 2) for _ in range(draws)])
    if comp.omega is not None:
        theory = f"{comp.omega * (g @ g):.4f}"            # exact
    elif isinstance(comp, RandK):
        theory = f"{(1 - comp.alpha) * (g @ g):.4f}"      # exact for unscaled Rand-k
    else:
        theory = f"<={(1 - comp.alpha) * (g @ g):.4f}"    # an upper bound for Top-k
    print(f"{comp.name:<20} {str(out):<38} {bits:>5} {err:>15.4f} {theory:>8}")

print("\nAverage of 100000 scaled Rand-2 outputs (should be close to g):")
comp = RandK(2)
comp.setup(len(g))
print(np.mean([comp(g, rng)[0] for _ in range(draws)], axis=0))
