"""Step 2 demo: uncompressed federated gradient descent on synthetic data.

    python scripts/run_baseline.py

Prints the loss, the squared gradient norm and the bits uploaded per device.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcomp import GD, LogisticRegression, run  # noqa: E402
from fedcomp.data import make_synthetic_classification, split_by_label  # noqa: E402

A, b = make_synthetic_classification(m=2000, d=50, seed=0)
As, bs = split_by_label(A, b, n=20)          # 20 devices with very different data
problem = LogisticRegression(As, bs, lam=0.1, reg="nonconvex")

L = problem.smoothness()
hist = run(problem, GD(), gamma=1.0 / L, rounds=500, log_every=50)

print(f"devices: {problem.n}   parameters: {problem.d}   L = {L:.4f}   stepsize = 1/L")
print(f"{'round':>6} {'loss':>10} {'|grad|^2':>12} {'Mbit/device':>12}")
for k, (f, g, bits) in enumerate(zip(hist.loss, hist.grad_norm_sq, hist.bits_per_device)):
    print(f"{k * 50:>6} {f:>10.6f} {g:>12.3e} {bits / 1e6:>12.3f}")
