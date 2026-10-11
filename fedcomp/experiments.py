"""Helpers for running fair comparisons.

The rule followed throughout: every method gets the same budget of uploaded
bits per device and the same stepsize search. The stepsize kept for a method
is the one that reaches the target accuracy with the fewest bits; if no
stepsize reaches it, the one with the lowest final error.
"""

from __future__ import annotations

import numpy as np

from .bits import dense_bits, randk_bits, topk_bits
from .compressors import RandK, TopK
from .methods import DCGD, DIANA, EF, EF21, GD
from .simulator import run

# key -> (label, compressor family). The order fixes each method's colour.
METHODS = {
    "gd":         ("GD (no compression)", None),
    "dcgd_top":   ("DCGD + Top-k", "top"),
    "ef_top":     ("EF + Top-k", "top"),
    "ef21_top":   ("EF21 + Top-k", "top"),
    "dcgd_rand":  ("DCGD + Rand-k", "rand"),
    "diana_rand": ("DIANA + Rand-k", "rand"),
}
RANDOM_METHODS = ("dcgd_rand", "diana_rand")
DEFAULT_MULTIPLIERS = tuple(2.0 ** j for j in range(-7, 3))   # stepsize = multiplier / L


def make_method(key: str, k: int):
    return {
        "gd": lambda: GD(),
        "dcgd_top": lambda: DCGD(TopK(k)),
        "ef_top": lambda: EF(TopK(k)),
        "ef21_top": lambda: EF21(TopK(k)),
        "dcgd_rand": lambda: DCGD(RandK(k)),
        "diana_rand": lambda: DIANA(RandK(k)),
    }[key]()


def bits_per_round(key: str, k: int, d: int) -> int:
    family = METHODS[key][1]
    if family is None:
        return dense_bits(d)
    return topk_bits(k, d) if family == "top" else randk_bits(k)


def bits_to_reach(bits, values, target):
    """Bits uploaded at the first logged point where values <= target (None if never)."""
    hit = np.flatnonzero(np.asarray(values) <= target)
    return float(bits[hit[0]]) if len(hit) else None


def tuned_run(problem, key, k, budget_bits, metric, target, seed=0,
              multipliers=DEFAULT_MULTIPLIERS, points=400):
    """Run one method at every stepsize in the grid and keep the best.

    metric: function x -> error (lower is better); target: the error level to reach.
    Returns a dict with the chosen stepsize multiplier and its curve.
    """
    L = problem.smoothness()
    rounds = max(1, int(budget_bits // bits_per_round(key, k, problem.d)))
    log_every = max(1, rounds // points)
    best = None
    for mult in multipliers:
        hist = run(problem, make_method(key, k), mult / L, rounds, seed=seed,
                   metrics={"err": metric}, log_every=log_every)
        if hist.diverged or not np.isfinite(hist.extra["err"][-1]):
            continue
        reach = bits_to_reach(hist.bits_per_device, hist.extra["err"], target)
        # fewest bits to the target first; otherwise lowest final error
        score = (0, reach) if reach is not None else (1, hist.extra["err"][-1])
        if best is None or score < best["score"]:
            best = {"score": score, "multiplier": mult, "rounds": rounds,
                    "bits": list(hist.bits_per_device), "err": list(hist.extra["err"]),
                    "bits_to_target": reach, "final_err": hist.extra["err"][-1]}
    return best


def repeat_run(problem, key, k, budget_bits, metric, multiplier, seeds, points=400):
    """Rerun a method at a fixed stepsize for several seeds; returns the mean curve."""
    L = problem.smoothness()
    rounds = max(1, int(budget_bits // bits_per_round(key, k, problem.d)))
    log_every = max(1, rounds // points)
    curves, bits = [], None
    for seed in seeds:
        hist = run(problem, make_method(key, k), multiplier / L, rounds, seed=seed,
                   metrics={"err": metric}, log_every=log_every)
        curves.append(hist.extra["err"])
        bits = list(hist.bits_per_device)
    curves = np.array(curves)
    return {"bits": bits, "err": list(curves.mean(axis=0)),
            "err_min": list(curves.min(axis=0)), "err_max": list(curves.max(axis=0))}
