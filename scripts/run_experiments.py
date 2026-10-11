"""Step 6: the experiments on the channel estimation task.

    python scripts/run_experiments.py            runs everything (several minutes)
    python scripts/run_experiments.py --quick    a small version to check that it runs

Experiments
  A  all methods at three compression levels (1%, 5%, 10% of coordinates kept)
  B  how the comparison changes as the users' data becomes more different
  C  EF21: the stepsize from its theorem against the best stepsize found

Every method gets the same budget of uploaded bits per device and the same
stepsize search. Results go to results/step6_results.json; make the figures
with scripts/plot_experiments.py.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcomp import EF21, TopK, dense_bits, ef21_stepsize, run  # noqa: E402
from fedcomp.channel import ChannelConfig, estimation_problem, make_dataset, test_nmse_metric, to_db  # noqa: E402
from fedcomp.experiments import (METHODS, RANDOM_METHODS, bits_per_round, bits_to_reach,  # noqa: E402
                                 repeat_run, tuned_run)

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

SNR_DB = 10.0
FRACTIONS = [0.01, 0.05, 0.10]
SPREADS = [0.0, 15.0, 30.0, 60.0]
DENSE_ROUNDS = 30          # budget = this many uncompressed uploads
TOLERANCE_DB = 0.1         # target: within this of the best error the model can reach
SEEDS = [0, 1, 2]
if args.quick:
    FRACTIONS, SPREADS, DENSE_ROUNDS, SEEDS = [0.05], [0.0, 30.0], 10, [0]


def setup(spread):
    cfg = ChannelConfig(snr_db=SNR_DB, center_spread_deg=spread)
    ds = make_dataset(cfg)
    problem = estimation_problem(ds)
    metric = test_nmse_metric(ds, problem)
    best = metric(problem.solution())            # error of the exact minimizer
    target = best * 10 ** (TOLERANCE_DB / 10)
    return problem, metric, best, target


def compare(problem, metric, target, fraction, label):
    d = problem.d
    k = max(1, round(fraction * d))
    budget = DENSE_ROUNDS * dense_bits(d)
    out = {}
    for key, (name, family) in METHODS.items():
        if family is None and fraction != FRACTIONS[0] and label == "A":
            continue                              # GD does not depend on k: run it once
        t0 = time.time()
        res = tuned_run(problem, key, k, budget, metric, target)
        if key in RANDOM_METHODS and len(SEEDS) > 1:   # average the random methods over seeds
            rep = repeat_run(problem, key, k, budget, metric, res["multiplier"], SEEDS)
            res.update(rep)
            res["bits_to_target"] = bits_to_reach(res["bits"], res["err"], target)
            res["final_err"] = res["err"][-1]
        res.pop("score")
        res["bits_per_round"] = bits_per_round(key, k, d)
        out[key] = res
        reach = "not reached" if res["bits_to_target"] is None else f"{res['bits_to_target'] / 1e6:8.3f} Mbit"
        print(f"  {name:<22} step {res['multiplier']:>9.5g}/L  final {to_db(res['final_err']):7.2f} dB"
              f"  to target: {reach}   ({time.time() - t0:.0f}s)", flush=True)
    return out


results = {"snr_db": SNR_DB, "tolerance_db": TOLERANCE_DB, "dense_rounds": DENSE_ROUNDS,
           "fractions": FRACTIONS, "spreads": SPREADS, "A": {}, "B": {}, "C": {}}

# ---- A: compression levels ---------------------------------------------------
problem, metric, best, target = setup(30.0)
results["d"], results["best_err"] = problem.d, best
print(f"parameters: {problem.d}   best reachable NMSE: {to_db(best):.2f} dB   "
      f"target: within {TOLERANCE_DB} dB   budget: {DENSE_ROUNDS} uncompressed uploads "
      f"= {DENSE_ROUNDS * dense_bits(problem.d) / 1e6:.2f} Mbit per device")
for f in FRACTIONS:
    print(f"\nA. keeping {100 * f:g}% of coordinates (k = {max(1, round(f * problem.d))})")
    results["A"][str(f)] = compare(problem, metric, target, f, "A")

# ---- C: EF21 theory stepsize against tuned -----------------------------------
print("\nC. EF21: stepsize from the theorem against the best found")
L = problem.smoothness()
for f in FRACTIONS:
    k = max(1, round(f * problem.d))
    gamma = ef21_stepsize(problem, k / problem.d)
    rounds = int(DENSE_ROUNDS * dense_bits(problem.d) // bits_per_round("ef21_top", k, problem.d))
    hist = run(problem, EF21(TopK(k)), gamma, rounds, metrics={"err": metric},
               log_every=max(1, rounds // 400))
    tuned = results["A"][str(f)]["ef21_top"]
    reach = bits_to_reach(hist.bits_per_device, hist.extra["err"], target)
    results["C"][str(f)] = {"theory_multiplier": gamma * L, "tuned_multiplier": tuned["multiplier"],
                            "theory_bits_to_target": reach, "tuned_bits_to_target": tuned["bits_to_target"],
                            "theory_final_err": hist.extra["err"][-1], "tuned_final_err": tuned["final_err"]}
    print(f"  {100 * f:>4g}%: theorem {gamma * L:.5f}/L, final {to_db(hist.extra['err'][-1]):.2f} dB, "
          f"to target {'not reached' if reach is None else f'{reach / 1e6:.3f} Mbit'}   |   "
          f"tuned {tuned['multiplier']:g}/L ({tuned['multiplier'] / (gamma * L):.0f} times larger)")

# ---- B: how different the users are -------------------------------------------
frac_b = 0.05
for spread in SPREADS:
    problem, metric, best, target = setup(spread)
    print(f"\nB. sector centres spread over {spread:g} degrees   best reachable NMSE: {to_db(best):.2f} dB")
    results["B"][str(spread)] = {"best_err": best, "methods": compare(problem, metric, target, frac_b, "B")}

os.makedirs("results", exist_ok=True)
name = "results/step6_results_quick.json" if args.quick else "results/step6_results.json"
with open(name, "w") as fh:
    json.dump(results, fh)
print(f"\nsaved {name}")
