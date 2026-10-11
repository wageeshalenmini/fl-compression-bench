"""Tests for step 4: DCGD, DIANA, classic error feedback and EF21.

Run with:  pytest -q
"""

import numpy as np
import pytest

from fedcomp import (DCGD, DIANA, EF, EF21, GD, Identity, LeastSquares,
                     LogisticRegression, RandK, TopK, ef21_stepsize, randk_bits,
                     run, topk_bits)
from fedcomp.data import make_synthetic_classification, split_by_label


# ---- problems used below ---------------------------------------------------

def logreg(n=10, m=600, d=30):
    A, b = make_synthetic_classification(m, d, seed=1)
    As, bs = split_by_label(A, b, n)              # devices with very different data
    return LogisticRegression(As, bs, lam=0.1, reg="nonconvex")


def hetero_least_squares(n=10, m=30, d=20):
    rng = np.random.default_rng(0)
    As = [rng.standard_normal((m, d)) for _ in range(n)]
    bs = [rng.standard_normal(m) + 3 * i for i in range(n)]   # targets differ by device
    return LeastSquares(As, bs)


def three_quadratics():
    """f_i(x) = <a_i, x>^2 + ||x||^2 / 4, the example on which Top-1 diverges
    (Beznosikov et al., "On Biased Compression for Distributed Learning").
    Written as least squares so that the existing problem class can be used."""
    As, bs = [], []
    for a in [(-3, 2, 2), (2, -3, 2), (2, 2, -3)]:
        a, m = np.array(a, dtype=float), 4
        As.append(np.vstack([np.sqrt(2 * m) * a, np.sqrt(m / 2) * np.eye(3)]))
        bs.append(np.zeros(m))
    return LeastSquares(As, bs)


# ---- with no compression, every method must be plain gradient descent -------

@pytest.mark.parametrize("make", [
    lambda: DCGD(Identity()),
    lambda: DIANA(Identity()),
    lambda: EF(Identity()),
    lambda: EF21(Identity()),
    lambda: DCGD(RandK(30)),
    lambda: DIANA(RandK(30)),
    lambda: EF(TopK(30)),
    lambda: EF21(TopK(30)),
])
def test_without_compression_every_method_equals_gd(make):
    prob = logreg()
    gamma, rounds = 0.5 / prob.smoothness(), 100
    ref = run(prob, GD(), gamma, rounds)
    out = run(prob, make(), gamma, rounds)
    np.testing.assert_allclose(out.x, ref.x, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(out.loss, ref.loss, rtol=1e-9)


# ---- DCGD ------------------------------------------------------------------

def test_dcgd_top1_diverges_on_three_quadratics():
    prob, t, gamma = three_quadratics(), 1.0, 0.05
    x0 = t * np.ones(3)
    # the local gradient at (t, t, t) is t * (-11/2, 9/2, 9/2) up to permutation
    np.testing.assert_allclose(prob.local_grad(0, x0), [-5.5, 4.5, 4.5])
    # so Top-1 keeps -11t/2 and one step multiplies x by (1 + 11 gamma / 6)
    one = run(prob, DCGD(TopK(1)), gamma, 1, x0=x0)
    np.testing.assert_allclose(one.x, t * (1 + 11 * gamma / 6) * np.ones(3))
    # which grows without bound for every positive stepsize
    many = run(prob, DCGD(TopK(1)), gamma, 400, x0=x0)
    assert many.diverged or many.loss[-1] > 1e6 * many.loss[0]
    small = run(prob, DCGD(TopK(1)), 1e-3, 400, x0=x0)
    assert small.loss[-1] > small.loss[0]


def test_dcgd_randk_stalls_at_a_floor_proportional_to_the_stepsize():
    prob = hetero_least_squares()
    base = 1.0 / (prob.smoothness() * (1 + 20 / 5 - 1))
    floor = {}
    for mult in (1.0, 0.25):
        hist = run(prob, DCGD(RandK(5)), mult * base, 4000)
        floor[mult] = np.mean(hist.grad_norm_sq[-1000:])
    assert floor[1.0] > 1e-1                      # it does not converge
    assert floor[0.25] == pytest.approx(floor[1.0] / 4, rel=0.25)


# ---- DIANA -----------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_diana_converges_exactly_where_dcgd_stalls(seed):
    prob = hetero_least_squares()
    gamma = 1.0 / (prob.smoothness() * (1 + 20 / 5 - 1))
    diana = run(prob, DIANA(RandK(5)), gamma, 4000, seed=seed)
    dcgd = run(prob, DCGD(RandK(5)), gamma, 4000, seed=seed)
    assert diana.grad_norm_sq[-1] < 1e-20
    np.testing.assert_allclose(diana.x, prob.solution(), atol=1e-8)
    assert np.mean(dcgd.grad_norm_sq[-1000:]) > 1.0


def test_diana_shifts_converge_to_local_gradients_at_the_solution():
    prob = hetero_least_squares()
    method = DIANA(RandK(5))
    run(prob, method, 1.0 / (prob.smoothness() * 4), 4000)
    x_star = prob.solution()
    for i in range(prob.n):
        np.testing.assert_allclose(method.h_dev[i], prob.local_grad(i, x_star), atol=1e-6)
    assert np.linalg.norm(prob.local_grad(0, x_star)) > 1.0   # and those are not zero


def test_diana_server_shift_equals_weighted_device_shifts():
    prob = hetero_least_squares()
    method = DIANA(RandK(5))

    def check(t, x, m):
        np.testing.assert_allclose(m.h, sum(w * h for w, h in zip(m.weights, m.h_dev)), atol=1e-10)

    run(prob, method, 0.05, 200, callback=check)


def test_diana_default_alpha_and_compressor_check():
    method = DIANA(RandK(5))
    method.setup(3, 20, np.ones(3) / 3)
    assert method.alpha == pytest.approx(1 / (1 + 3.0))
    with pytest.raises(ValueError):
        DIANA(TopK(5)).setup(3, 20, np.ones(3) / 3)


# ---- classic error feedback --------------------------------------------------

def test_ef_virtual_iterate_follows_exact_gradient_descent():
    """x_t - sum_i p_i e_i^t moves by exactly -gamma * grad F(x_t) each round."""
    prob, gamma = logreg(), 0.05
    method = EF(TopK(3))
    seen = []

    def record(t, x, m):
        virtual = x - sum(w * e for w, e in zip(m.weights, m.e))
        seen.append((virtual, prob.grad(x)))

    run(prob, method, gamma, 60, callback=record)
    for (v_now, g_now), (v_next, _) in zip(seen[:-1], seen[1:]):
        np.testing.assert_allclose(v_next, v_now - gamma * g_now, atol=1e-12)


def test_ef_converges_on_three_quadratics_where_dcgd_diverges():
    prob = three_quadratics()
    hist = run(prob, EF(TopK(1)), ef21_stepsize(prob, 1 / 3), 3000, x0=np.ones(3))
    assert hist.grad_norm_sq[-1] < 1e-12


# ---- EF21 ------------------------------------------------------------------

def test_ef21_first_message_is_the_compressed_gradient():
    prob = logreg()
    method = EF21(TopK(3))
    run(prob, method, 0.01, 1)
    x0 = np.zeros(prob.d)
    for i in range(prob.n):
        expected, _ = TopK(3)(prob.local_grad(i, x0))
        np.testing.assert_allclose(method.g_dev[i], expected)


def test_ef21_server_estimate_equals_weighted_device_estimates():
    """The sync property that the convergence proof relies on."""
    prob = logreg()
    method = EF21(TopK(3))

    def check(t, x, m):
        np.testing.assert_allclose(m.g, sum(w * g for w, g in zip(m.weights, m.g_dev)), atol=1e-12)

    run(prob, method, 0.02, 300, callback=check)


def test_ef21_converges_on_three_quadratics_where_dcgd_diverges():
    prob = three_quadratics()
    hist = run(prob, EF21(TopK(1)), ef21_stepsize(prob, 1 / 3), 3000, x0=np.ones(3))
    assert hist.grad_norm_sq[-1] < 1e-12
    np.testing.assert_allclose(hist.x, 0, atol=1e-6)


@pytest.mark.parametrize("k", [1, 3, 10])
def test_ef21_satisfies_the_bound_of_its_theorem(k):
    """Theorem 1 of the EF21 paper, with f_inf replaced by 0 (the loss is >= 0):
    average of ||grad F(x_t)||^2 over t < T  <=  2 F(x_0)/(gamma T) + G_0/(theta T)."""
    prob, T = logreg(), 2000
    alpha = k / prob.d
    gamma = ef21_stepsize(prob, alpha)
    hist = run(prob, EF21(TopK(k)), gamma, T)

    x0 = np.zeros(prob.d)
    G0 = sum(prob.p[i] * np.sum((TopK(k)(prob.local_grad(i, x0))[0] - prob.local_grad(i, x0)) ** 2)
             for i in range(prob.n))
    theta = 1 - np.sqrt(1 - alpha)
    bound = 2 * hist.loss[0] / (gamma * T) + G0 / (theta * T)

    assert np.mean(hist.grad_norm_sq[:T]) <= bound
    assert hist.grad_norm_sq[-1] < 1e-3 * hist.grad_norm_sq[0]


def test_ef21_estimates_converge_to_the_local_gradients():
    prob = logreg()
    method = EF21(TopK(10))
    hist = run(prob, method, ef21_stepsize(prob, 10 / prob.d), 2000)
    for i in range(prob.n):
        np.testing.assert_allclose(method.g_dev[i], prob.local_grad(i, hist.x), atol=1e-8)


def test_ef21_stepsize_formula():
    prob = logreg()
    assert ef21_stepsize(prob, 1.0) == pytest.approx(1 / prob.smoothness())
    assert ef21_stepsize(prob, 0.1) < ef21_stepsize(prob, 0.5) < ef21_stepsize(prob, 1.0)
    # alpha = 3/4: theta = 1/2, beta = 1/2, so gamma = 1 / (L + L_tilde)
    L_tilde = np.sqrt(sum(prob.p[i] * prob.local_smoothness(i) ** 2 for i in range(prob.n)))
    assert ef21_stepsize(prob, 0.75) == pytest.approx(1 / (prob.smoothness() + L_tilde))


@pytest.mark.parametrize("make", [lambda: EF21(RandK(5)), lambda: EF(RandK(5))])
def test_error_feedback_methods_reject_non_contractive_compressors(make):
    with pytest.raises(ValueError):
        make().setup(3, 20, np.ones(3) / 3)


def test_ef21_accepts_unscaled_randk():
    prob = logreg()
    hist = run(prob, EF21(RandK(5, scaled=False)), ef21_stepsize(prob, 5 / prob.d), 3000)
    assert hist.grad_norm_sq[-1] < 1e-6 * hist.grad_norm_sq[0]


# ---- bits and randomness -----------------------------------------------------

@pytest.mark.parametrize("make, per_round", [
    (lambda: DCGD(RandK(5)), randk_bits(5)),
    (lambda: DIANA(RandK(5)), randk_bits(5)),
    (lambda: EF(TopK(3)), topk_bits(3, 30)),
    (lambda: EF21(TopK(3)), topk_bits(3, 30)),
])
def test_bits_per_device_per_round(make, per_round):
    hist = run(logreg(), make(), 0.01, 25)
    assert hist.bits_per_device[-1] == 25 * per_round


def test_same_seed_same_run_different_seed_different_run():
    prob = logreg()
    a = run(prob, DIANA(RandK(5)), 0.02, 50, seed=3)
    b = run(prob, DIANA(RandK(5)), 0.02, 50, seed=3)
    c = run(prob, DIANA(RandK(5)), 0.02, 50, seed=4)
    np.testing.assert_array_equal(a.x, b.x)
    assert not np.allclose(a.x, c.x)


def test_devices_draw_independent_masks():
    method = DCGD(RandK(3))
    method.setup(4, 50, np.ones(4) / 4, seed=0)
    v = np.ones(50)
    masks = [tuple(np.flatnonzero(method.device_step(i, v, 0.1).payload)) for i in range(4)]
    assert len(set(masks)) == 4
