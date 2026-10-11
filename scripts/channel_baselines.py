"""Step 5 check: channel estimation error against SNR.

    python scripts/channel_baselines.py

Compares, on the default system (64 antennas, 32 pilots, 20 users):
  - least squares (no prior knowledge)
  - one LMMSE estimator shared by all users, from the true covariance
  - one LMMSE estimator per user, from that user's true covariance
  - the shared estimator trained by federated gradient descent on samples

Writes results/step5_baselines.png and prints the table behind it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fedcomp import GD, run  # noqa: E402
from fedcomp.channel import (ChannelConfig, estimation_problem, linear_mse,  # noqa: E402
                             lmmse_estimator, ls_estimator, make_dataset,
                             test_nmse_metric, to_db)

SNRS = [-10, -5, 0, 5, 10, 15, 20]
ROUNDS = 600
rows = []

cfg0 = ChannelConfig()
print(f"antennas: {cfg0.n_antennas}   pilots: {cfg0.n_pilots}   users: {cfg0.n_users}   "
      f"training pairs per user: {cfg0.m_train}")
print(f"sector width: {cfg0.sector_width_deg:g} deg   centres spread over: {cfg0.center_spread_deg:g} deg\n")
print(f"{'SNR dB':>6} {'LS':>8} {'shared LMMSE':>13} {'per-user LMMSE':>15} {'federated GD':>13}   (NMSE in dB)")

for snr in SNRS:
    cfg = ChannelConfig(snr_db=snr)
    ds = make_dataset(cfg)
    N, s2 = cfg.n_antennas, cfg.sigma2
    ls = linear_mse(ls_estimator(ds.P), ds.P, ds.R_mix, s2) / N
    shared = linear_mse(lmmse_estimator(ds.P, ds.R_mix, s2), ds.P, ds.R_mix, s2) / N
    per_user = np.mean([linear_mse(lmmse_estimator(ds.P, R, s2), ds.P, R, s2) for R in ds.R]) / N

    problem = estimation_problem(ds)
    hist = run(problem, GD(), 1.0 / problem.smoothness(), ROUNDS, log_every=ROUNDS)
    trained = test_nmse_metric(ds, problem)(hist.x)

    rows.append((snr, to_db(ls), to_db(shared), to_db(per_user), to_db(trained)))
    print(f"{snr:>6} {rows[-1][1]:>8.2f} {rows[-1][2]:>13.2f} {rows[-1][3]:>15.2f} {rows[-1][4]:>13.2f}")

# ---- plot ------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

r = np.array(rows)
fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
ax.plot(r[:, 0], r[:, 1], color="#2a78d6", linestyle="--", linewidth=2, marker="s", markersize=6,
        label="Least squares")
ax.plot(r[:, 0], r[:, 2], color="#eb6834", linestyle="-", linewidth=2,
        label="Shared LMMSE (true covariance)")
ax.plot(r[:, 0], r[:, 4], color="#1baf7a", linestyle="none", marker="o", markersize=8,
        markerfacecolor="none", markeredgewidth=2, label="Shared estimator, federated GD")
ax.plot(r[:, 0], r[:, 3], color="#eda100", linestyle="-.", linewidth=2, marker="^", markersize=6,
        label="Per-user LMMSE (true covariance)")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("NMSE (dB)")
ax.set_title(f"Channel estimation error, {cfg0.n_antennas} antennas, {cfg0.n_pilots} pilots, "
             f"{cfg0.n_users} users", fontsize=11, loc="left")
ax.grid(True, color="#e6e5e0", linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/step5_baselines.png")
print("\nsaved results/step5_baselines.png")
