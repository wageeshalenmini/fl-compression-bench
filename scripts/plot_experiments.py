"""Figures and tables for step 6, from results/step6_results.json.

    python scripts/plot_experiments.py
    python scripts/plot_experiments.py --quick     uses the quick results file
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fedcomp.experiments import METHODS  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
suffix = "_quick" if args.quick else ""
with open(f"results/step6_results{suffix}.json") as fh:
    R = json.load(fh)

# one fixed colour and line style per method, the same in every figure
STYLE = {
    "gd":         ("#2a78d6", "-"),
    "dcgd_top":   ("#eb6834", "--"),
    "ef_top":     ("#1baf7a", "-."),
    "ef21_top":   ("#eda100", "-"),
    "dcgd_rand":  ("#e87ba4", ":"),
    "diana_rand": ("#008300", "--"),
}
GRID = "#e6e5e0"


def db(x):
    return 10 * np.log10(np.asarray(x, dtype=float))


def mbit(v):
    return "not reached" if v is None else f"{v / 1e6:.3f}"


def clean(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


best_db = float(db(R["best_err"]))
fractions = [str(f) for f in R["fractions"]]
gd = R["A"][fractions[0]]["gd"]

# ---- table A ---------------------------------------------------------------
print(f"A. Mbit uploaded per device to get within {R['tolerance_db']} dB of the best error "
      f"({best_db:.2f} dB), and final error in dB\n")
header = f"{'method':<22}" + "".join(f"{f'keep {100 * float(f):g}%':>26}" for f in fractions)
print(header)
for key, (name, family) in METHODS.items():
    row = f"{name:<22}"
    for f in fractions:
        res = gd if key == "gd" else R["A"][f][key]
        row += f"{mbit(res['bits_to_target']):>14} {float(db(res['final_err'])):>8.2f} dB"
    print(row)

# ---- figure A: one panel per compression level ------------------------------
fig, axes = plt.subplots(1, len(fractions), figsize=(4.6 * len(fractions), 4.6), dpi=150,
                         sharey=True, squeeze=False)
for ax, f in zip(axes[0], fractions):
    for key, (name, family) in METHODS.items():
        res = gd if key == "gd" else R["A"][f][key]
        colour, style = STYLE[key]
        x = np.array(res["bits"][1:]) / 1e6
        ax.plot(x, db(res["err"][1:]), color=colour, linestyle=style, linewidth=2, label=name)
    ax.axhline(best_db, color="#52514e", linewidth=1, linestyle=(0, (1, 2)))
    ax.set_xscale("log")
    ax.set_ylim(best_db - 0.4, 0.3)
    ax.set_xlabel("Mbit uploaded per device")
    ax.set_title(f"keep {100 * float(f):g}% of coordinates", fontsize=10, loc="left")
    clean(ax)
axes[0][0].set_ylabel("test NMSE (dB)")
axes[0][0].annotate("best reachable", xy=(axes[0][0].get_xlim()[0], best_db), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color="#52514e")
axes[0][-1].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("Channel estimator: error against uploaded bits, at equal budget", fontsize=11, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(f"results/step6_main{suffix}.png")

# ---- table and figure B: how different the users are -------------------------
print("\nB. Effect of how different the users are (keeping 5% of coordinates)\n")
spreads = [str(s) for s in R["spreads"]]
print(f"{'method':<22}" + "".join(f"{f'spread {float(s):g} deg':>26}" for s in spreads))
print(f"{'best reachable':<22}" + "".join(f"{float(db(R['B'][s]['best_err'])):>23.2f} dB" for s in spreads))
for key, (name, family) in METHODS.items():
    row = f"{name:<22}"
    for s in spreads:
        res = R["B"][s]["methods"][key]
        gap = float(db(res["final_err"]) - db(R["B"][s]["best_err"]))
        row += f"{mbit(res['bits_to_target']):>14} {gap:>+8.2f} dB"
    print(row)
print("(columns: Mbit to target, and final gap to the best reachable error)")

fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
xs = [float(s) for s in spreads]
budget = max(R["B"][spreads[0]]["methods"]["gd"]["bits"]) / 1e6
for key, (name, family) in METHODS.items():
    colour, style = STYLE[key]
    ys = [R["B"][s]["methods"][key]["bits_to_target"] for s in spreads]
    ys = [np.nan if y is None else y / 1e6 for y in ys]      # a gap in the line = not reached
    ax.plot(xs, ys, color=colour, linestyle=style, linewidth=2, marker="o", markersize=6, label=name)
ax.axhline(budget, color="#52514e", linewidth=1, linestyle=(0, (1, 2)))
ax.annotate("budget (points missing above it were not reached)", xy=(xs[0], budget), xytext=(0, 4),
            textcoords="offset points", fontsize=8, color="#52514e")
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xlabel("spread of the users' sector centres (degrees); 0 means all users alike")
ax.set_ylabel(f"Mbit per device to get within {R['tolerance_db']} dB of best")
ax.set_title("Bits needed as the users' data becomes more different (keep 5%)",
             fontsize=11, loc="left")
clean(ax)
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(f"results/step6_heterogeneity{suffix}.png")

# ---- table C -----------------------------------------------------------------
print("\nC. EF21: stepsize from the theorem against the best stepsize found\n")
print(f"{'kept':>6} {'theorem':>12} {'tuned':>9} {'ratio':>7} {'theorem: Mbit to target':>25} {'tuned: Mbit to target':>23}")
for f in fractions:
    c = R["C"][f]
    print(f"{100 * float(f):>5g}% {c['theory_multiplier']:>10.5f}/L {c['tuned_multiplier']:>7g}/L "
          f"{c['tuned_multiplier'] / c['theory_multiplier']:>7.0f} {mbit(c['theory_bits_to_target']):>25} "
          f"{mbit(c['tuned_bits_to_target']):>23}")

print(f"\nsaved results/step6_main{suffix}.png and results/step6_heterogeneity{suffix}.png")
