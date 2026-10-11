"""Tests for step 2: the core simulator with no compression.

Run with:  pytest -q
"""

import numpy as np
import pytest

from fedcomp import GD, LeastSquares, LogisticRegression, run, dense_bits
from fedcomp.data import make_synthetic_classification, split_by_label, split_iid


def finite_difference_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for k in range(len(x)):
        e = np.zeros_like(x)
        e[k] = eps
        g[k] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


def make_logreg(n=7, m=350, d=12, reg="nonconvex", unequal=True):
    A, b = make_synthetic_classification(m, d, seed=1)
    if unequal:  # devices with different amounts of data and different label mixes
        idx = np.argsort(b, kind="stable")
        cuts = np.sort(np.random.default_rng(2).choice(np.arange(20, m - 20), n - 1, replace=False))
        parts = np.split(idx, cuts)
        As, bs = [A[p] for p in parts], [b[p] for p in parts]
    else:
        As, bs = split_iid(A, b, n)
    return LogisticRegression(As, bs, lam=0.1, reg=reg)


# ---- the worked example from the study notes (Part 1) --------------------

def notes_example():
    data = [[(1, 3), (2, 5)], [(0, 1), (1, 3)], [(2, 5), (3, 7)], [(1, 4), (2, 6)], [(0, 0), (2, 4)]]
    As = [np.array([[x, 1.0] for x, _ in pts]) for pts in data]
    bs = [np.array([y for _, y in pts], dtype=float) for pts in data]
    return LeastSquares(As, bs)


def test_hand_computed_round():
    prob = notes_example()
    x0 = np.zeros(2)
    np.testing.assert_allclose(prob.local_grad(0, x0), [-6.5, -4.0])
    np.testing.assert_allclose(prob.local_grad(2, x0), [-15.5, -6.0])
    np.testing.assert_allclose(prob.grad(x0), [-7.1, -3.8])
    assert prob.loss(x0) == pytest.approx(9.3)

    hist = run(prob, GD(), gamma=0.1, rounds=1)
    np.testing.assert_allclose(hist.x, [0.71, 0.38])
    assert hist.loss[-1] == pytest.approx(3.97066)


# ---- gradients are correct ------------------------------------------------

@pytest.mark.parametrize("reg", ["nonconvex", "l2"])
def test_logreg_gradient_matches_finite_differences(reg):
    prob = make_logreg(reg=reg)
    x = np.random.default_rng(0).standard_normal(prob.d)
    for i in range(prob.n):
        fd = finite_difference_grad(lambda v: prob.local_loss(i, v), x)
        np.testing.assert_allclose(prob.local_grad(i, x), fd, rtol=1e-5, atol=1e-7)
    fd = finite_difference_grad(prob.loss, x)
    np.testing.assert_allclose(prob.grad(x), fd, rtol=1e-5, atol=1e-7)


def test_least_squares_gradient_matches_finite_differences():
    prob = notes_example()
    x = np.array([0.3, -1.2])
    np.testing.assert_allclose(prob.grad(x), finite_difference_grad(prob.loss, x), rtol=1e-6)


# ---- the key test: federated GD equals GD on the pooled data ---------------

@pytest.mark.parametrize("unequal", [True, False])
def test_federated_gd_equals_pooled_gd(unequal):
    prob = make_logreg(unequal=unequal)
    pooled = prob.pooled()
    assert pooled.n == 1 and pooled.m[0] == prob.m.sum()

    gamma, rounds = 0.5 / prob.smoothness(), 200
    fed = run(prob, GD(), gamma, rounds)
    cen = run(pooled, GD(), gamma, rounds)

    np.testing.assert_allclose(fed.x, cen.x, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fed.loss, cen.loss, rtol=1e-10)


def test_weights_follow_data_sizes():
    prob = make_logreg(unequal=True)
    assert len(set(prob.m)) > 1                      # sizes really differ
    np.testing.assert_allclose(prob.p, prob.m / prob.m.sum())
    assert prob.p.sum() == pytest.approx(1.0)


# ---- gradient descent behaves as theory says -------------------------------

def test_gd_decreases_loss_monotonically_with_step_1_over_L():
    prob = make_logreg()
    hist = run(prob, GD(), 1.0 / prob.smoothness(), 300)
    assert np.all(np.diff(hist.loss) <= 1e-12)
    assert hist.grad_norm_sq[-1] < 1e-3 * hist.grad_norm_sq[0]


def test_gd_reaches_least_squares_solution():
    rng = np.random.default_rng(3)
    As = [rng.standard_normal((30, 5)) for _ in range(4)]
    bs = [rng.standard_normal(30) for _ in range(4)]
    prob = LeastSquares(As, bs)
    hist = run(prob, GD(), 1.0 / prob.smoothness(), 2000)
    np.testing.assert_allclose(hist.x, prob.solution(), atol=1e-8)


def test_smoothness_bound_holds():
    prob = make_logreg()
    rng = np.random.default_rng(4)
    L = prob.smoothness()
    for _ in range(50):
        x, y = rng.standard_normal(prob.d), rng.standard_normal(prob.d)
        assert np.linalg.norm(prob.grad(x) - prob.grad(y)) <= L * np.linalg.norm(x - y) * (1 + 1e-9)


# ---- bit accounting --------------------------------------------------------

def test_bit_counter_for_uncompressed_gd():
    prob = make_logreg()
    rounds = 17
    hist = run(prob, GD(), 0.1, rounds)
    assert hist.bits_per_device[0] == 0
    assert hist.bits_per_device[-1] == rounds * dense_bits(prob.d) == rounds * 32 * prob.d
    assert len(hist.loss) == rounds + 1


# ---- misc ------------------------------------------------------------------

def test_divergence_is_detected():
    prob = notes_example()
    hist = run(prob, GD(), gamma=50.0, rounds=500)
    assert hist.diverged


def test_custom_metric_is_logged():
    prob = notes_example()
    target = prob.solution()
    hist = run(prob, GD(), 0.1, 10, metrics={"dist": lambda x: np.linalg.norm(x - target)})
    assert len(hist.extra["dist"]) == 11
    assert hist.extra["dist"][-1] < hist.extra["dist"][0]


def test_split_by_label_makes_devices_differ():
    A, b = make_synthetic_classification(400, 6, seed=0)
    _, bs = split_by_label(A, b, 4)
    assert np.all(bs[0] == -1) and np.all(bs[-1] == 1)
