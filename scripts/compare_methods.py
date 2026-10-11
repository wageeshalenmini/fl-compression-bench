"""Step 4 check: all methods on one problem, at an equal budget of uploaded bits.

    python scripts/compare_methods.py                  synthetic data
    python scripts/compare_methods.py --data a9a       a LibSVM file you downloaded

Setup follows the EF21 paper: logistic regression with a nonconvex
regularizer (lambda = 0.1), 20 devices holding very different data, Top-1.
Every method gets the same stepsize search: 1/L times a power of two, and
the best non-diverging run (lowest final squared gradient norm) is kept.

Writes results/step4_comparison.png and prints the table behind it.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcomp import (DCGD, DIANA, EF, EF21, GD, LogisticRegression, RandK,  # noqa: E402
                     TopK, dense_bits, randk_bits, run, topk_bits)
from fedcomp.data import load_libsvm, make_synthetic_classification, split_by_label  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None, help="path to a LibSVM file; synthetic data if omitted")
parser.add_argument("--devices", type=int, default=20)
parser.add_argument("--budget", type=float, default=1e5, help="uplink bits per device")
args = parser.parse_args()

if args.data:
    A, b = load_libsvm(args.data)
    data_name = os.path.basename(args.data)
else:
    A, b = make_synthetic_classification(m=2000, d=50, seed=0)
    data_name = "synthetic"
As, bs = split_by_label(A, b, args.devices)
problem = LogisticRegression(As, bs, lam=0.1, reg="nonconvex")
d, L = problem.d, problem.smoothness()

# name, constructor, bits per device per round, colour, line style
METHODS = [
    ("GD (no compression)", lambda: GD(),            dense_bits(d),   "#2a78d6", "-"),
    ("DCGD + Top-1",        lambda: DCGD(TopK(1)),   topk_bits(1, d), "#eb6834", "--"),
    ("EF + Top-1",          lambda: EF(TopK(1)),     topk_bits(1, d), "#1baf7a", "-."),
    ("EF21 + Top-1",        lambda: EF21(TopK(1)),   topk_bits(1, d), "#eda100", "-"),
    ("DCGD + Rand-2",       lambda: DCGD(RandK(2)),  randk_bits(2),   "#e87ba4", ":"),
    ("DIANA + Rand-2",      lambda: DIANA(RandK(2)), randk_bits(2),   "#008300", "--"),
]
MULTIPLIERS = [2.0 ** j for j in range(-7, 3)]

print(f"data: {data_name}   devices: {problem.n}   parameters: {d}   L = {L:.4f}")
print(f"budget: {args.budget:.0f} bits per device\n")
print(f"{'method':<22} {'bits/round':>10} {'rounds':>7} {'best step':>10} {'final |grad|^2':>15} {'final loss':>11}")

best = {}
for name, make, bits_per_round, _, _ in METHODS:
    rounds = int(args.budget // bits_per_round)
    log_every = max(1, rounds // 400)
    for mult in MULTIPLIERS:
        hist = run(problem, make(), mult / L, rounds, log_every=log_every)
        if hist.diverged or not np.isfinite(hist.grad_norm_sq[-1]):
            continue
        if name not in best or hist.grad_norm_sq[-1] < best[name][1].grad_norm_sq[-1]:
            best[name] = (mult, hist)
    mult, hist = best[name]
    print(f"{name:<22} {bits_per_round:>10} {rounds:>7} {f'{mult:g}/L':>10} "
          f"{hist.grad_norm_sq[-1]:>15.3e} {hist.loss[-1]:>11.6f}")

# ---- plot ------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
for name, _, _, colour, style in METHODS:
    hist = best[name][1]
    x = np.array(hist.bits_per_device) / 1e3
    y = np.maximum(np.array(hist.grad_norm_sq), 1e-32)
    ax.plot(x, y, color=colour, linestyle=style, linewidth=2, label=name)
ax.set_yscale("log")
ax.set_xlabel("bits uploaded per device (thousands)")
ax.set_ylabel("squared gradient norm")
ax.set_title(f"Compression methods at equal uplink budget ({data_name}, {problem.n} devices, d = {d})",
             fontsize=11, loc="left")
ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/step4_comparison.png")
print("\nsaved results/step4_comparison.png")
