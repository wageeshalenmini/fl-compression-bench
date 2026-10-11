# fl-compression-bench

A small, tested simulator for comparing gradient compression methods in
federated learning, with federated channel estimation as the target task.

The question: for a federated channel estimator, which compression method
reaches a given estimation accuracy with the fewest uploaded bits, and do the
methods behave as their theory predicts?

This is a benchmark study. It does not propose a new method.

## Status

| Step | Content | State |
| --- | --- | --- |
| 1 | Reading: update rules and theoretical stepsizes of each method | in progress |
| 2 | Core simulator: devices, server, full local gradients, bit counter | done |
| 3 | Compressors: Rand-k, Top-k | to do |
| 4 | Methods: compressed GD, DIANA, classic error feedback, EF21 | to do |
| 5 | Channel estimation task with least squares and LMMSE baselines | to do |
| 6 | Experiments | to do |
| 7 | Report | to do |

## Layout

```
fedcomp/
  problems.py   LeastSquares and LogisticRegression as sums over devices
  methods.py    Method interface (device_step / server_step) and uncompressed GD
  simulator.py  the training loop and the History record
  bits.py       bit accounting for uplink messages
  data.py       synthetic data, LibSVM loader, IID and by-label splits
tests/
  test_core.py  tests for step 2
scripts/
  run_baseline.py  demo run of uncompressed federated GD
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate        (Windows)   or   source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/run_baseline.py
```

## Design

One round of training:

1. the server broadcasts the model `x`
2. each device computes its local gradient at `x`
3. each device runs `method.device_step` and uploads a `Message` (payload and bit cost)
4. the server runs `method.server_step` and updates `x`

A method is split into `device_step` and `server_step` so that the code
mirrors where computation happens: device code sees only its own gradient
and state, server code sees only messages. Both receive the stepsize and the
server returns the step to add to `x`, which lets the same interface cover
plain gradient descent, DIANA, classic error feedback and EF21.

Device `i` holds `m_i` points and has weight `p_i = m_i / sum_j m_j`. The
objective is `F(x) = sum_i p_i F_i(x)`. Only uplink bits are counted, with a
real number costing 32 bits.

## What the step 2 tests check

- the hand-computed round from the study notes: five devices, two parameters,
  one step from (0, 0) to (0.71, 0.38), loss 9.3 to 3.97
- gradients against finite differences
- federated GD equals GD on the pooled data to 1e-10, with equal and unequal
  device sizes
- GD with stepsize 1/L decreases the loss in every round
- GD reaches the exact least squares solution
- the smoothness bound holds on random pairs of points
- the bit counter: `rounds * 32 * d` per device for uncompressed GD

## References

- Konečný, McMahan, Yu, Richtárik, Suresh, Bacon. Federated Learning: Strategies for Improving Communication Efficiency. arXiv:1610.05492
- Mishchenko, Gorbunov, Takáč, Richtárik. Distributed Learning with Compressed Gradient Differences (DIANA). arXiv:1901.09269
- Richtárik, Sokolov, Fatkhullin. EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback. arXiv:2106.05203
