"""Tests for step 5: the channel model, the classical estimators and the
federated estimation problem.

Run with:  pytest -q
"""

import numpy as np
import pytest

from fedcomp import EF21, GD, LinearEstimation, TopK, run
from fedcomp.channel import (ChannelConfig, estimation_problem, linear_mse,
                             lmmse_estimator, ls_estimator, make_dataset,
                             make_pilots, nmse, observe, real_matrix,
                             sample_channels, steering, test_nmse_metric,
                             to_db, to_real, user_covariance)

SMALL = dict(n_antennas=16, n_pilots=8, n_users=6, m_train=400, m_test=4000)


def pooled(pairs):
    return np.vstack([Y for Y, _ in pairs]), np.vstack([H for _, H in pairs])


# ---- channel model ---------------------------------------------------------

def test_steering_vectors_have_unit_norm():
    a = steering(64, np.linspace(-1.2, 1.2, 50))
    np.testing.assert_allclose(np.linalg.norm(a, axis=-1), 1.0)
    np.testing.assert_allclose(steering(8, 0.0), np.ones(8) / np.sqrt(8))   # broadside


def test_pilot_rows_have_unit_norm():
    P = make_pilots(64, 32, np.random.default_rng(0))
    assert P.shape == (32, 64)
    np.testing.assert_allclose(np.linalg.norm(P, axis=1), 1.0)


def test_channel_power_is_n_antennas():
    H = sample_channels(20000, 16, 3, 0.2, 0.3, np.random.default_rng(0))
    assert np.mean(np.sum(np.abs(H) ** 2, axis=1)) == pytest.approx(16, rel=0.03)


def test_exact_covariance_matches_sample_covariance():
    N, center, width = 12, -0.3, 0.25
    R = user_covariance(N, center, width)
    np.testing.assert_allclose(R, R.conj().T, atol=1e-12)          # Hermitian
    assert np.real(np.trace(R)) == pytest.approx(N)                 # power N
    H = sample_channels(200000, N, 3, center, width, np.random.default_rng(1))
    R_hat = H.T @ H.conj() / len(H)
    assert np.linalg.norm(R_hat - R) / np.linalg.norm(R) < 0.02


def test_noise_has_the_configured_power():
    P = make_pilots(16, 8, np.random.default_rng(0))
    Y = observe(np.zeros((50000, 16), dtype=complex), P, 0.25, np.random.default_rng(1))
    assert np.mean(np.abs(Y) ** 2) == pytest.approx(0.25, rel=0.02)


def test_snr_conversion():
    assert ChannelConfig(snr_db=10).sigma2 == pytest.approx(0.1)
    assert ChannelConfig(snr_db=-10).sigma2 == pytest.approx(10.0)
    assert to_db(0.01) == pytest.approx(-20.0)


def test_same_sector_for_everyone_when_spread_is_zero():
    ds = make_dataset(ChannelConfig(center_spread_deg=0.0, **SMALL))
    for R in ds.R[1:]:
        np.testing.assert_allclose(R, ds.R[0])
    spread = make_dataset(ChannelConfig(center_spread_deg=60.0, **SMALL))
    assert np.linalg.norm(spread.R[0] - spread.R[-1]) > 1.0


# ---- classical estimators --------------------------------------------------

@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_error_formula_matches_simulation(snr_db):
    """linear_mse is exact; the simulated error must agree with it."""
    cfg = ChannelConfig(snr_db=snr_db, **SMALL)
    ds = make_dataset(cfg)
    Y, H = pooled(ds.test)
    for W in (ls_estimator(ds.P), lmmse_estimator(ds.P, ds.R_mix, cfg.sigma2)):
        theory = linear_mse(W, ds.P, ds.R_mix, cfg.sigma2) / cfg.n_antennas
        assert nmse(W, Y, H) == pytest.approx(theory, rel=0.05)


def test_ls_error_formula_for_fewer_pilots_than_antennas():
    """LS error = channel energy outside the pilot row space + amplified noise."""
    cfg = ChannelConfig(**SMALL)
    ds = make_dataset(cfg)
    P, R = ds.P, ds.R_mix
    proj = np.linalg.pinv(P) @ P
    unseen = np.real(np.trace((np.eye(cfg.n_antennas) - proj) @ R))
    noise = cfg.sigma2 * np.real(np.trace(np.linalg.inv(P @ P.conj().T)))
    assert linear_mse(ls_estimator(P), P, R, cfg.sigma2) == pytest.approx(unseen + noise)


def test_lmmse_is_the_best_linear_estimator():
    cfg = ChannelConfig(**SMALL)
    ds = make_dataset(cfg)
    W = lmmse_estimator(ds.P, ds.R_mix, cfg.sigma2)
    best = linear_mse(W, ds.P, ds.R_mix, cfg.sigma2)
    assert best < linear_mse(ls_estimator(ds.P), ds.P, ds.R_mix, cfg.sigma2)
    rng = np.random.default_rng(0)
    for _ in range(20):                       # any perturbation makes it worse
        dW = 0.05 * (rng.standard_normal(W.shape) + 1j * rng.standard_normal(W.shape))
        assert linear_mse(W + dW, ds.P, ds.R_mix, cfg.sigma2) > best


@pytest.mark.parametrize("snr_db", [-10.0, 0.0, 10.0, 20.0])
def test_ordering_ls_shared_lmmse_per_user_lmmse(snr_db):
    """Knowing more helps: LS >= one LMMSE for all users >= one LMMSE per user."""
    cfg = ChannelConfig(snr_db=snr_db, m_train=10, m_test=10)      # default system
    ds = make_dataset(cfg)
    s2 = cfg.sigma2
    ls = linear_mse(ls_estimator(ds.P), ds.P, ds.R_mix, s2)
    shared = linear_mse(lmmse_estimator(ds.P, ds.R_mix, s2), ds.P, ds.R_mix, s2)
    per_user = np.mean([linear_mse(lmmse_estimator(ds.P, R, s2), ds.P, R, s2) for R in ds.R])
    assert per_user < shared < ls


def test_default_system_reference_values_at_10_db():
    cfg = ChannelConfig(m_train=10, m_test=10)
    ds = make_dataset(cfg)
    N, s2 = cfg.n_antennas, cfg.sigma2
    assert to_db(linear_mse(ls_estimator(ds.P), ds.P, ds.R_mix, s2) / N) == pytest.approx(-2.26, abs=0.05)
    assert to_db(linear_mse(lmmse_estimator(ds.P, ds.R_mix, s2), ds.P, ds.R_mix, s2) / N) \
        == pytest.approx(-8.30, abs=0.05)


# ---- real / complex conversion ----------------------------------------------

def test_real_matrix_acts_like_the_complex_matrix():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((5, 3)) + 1j * rng.standard_normal((5, 3))
    y = rng.standard_normal((7, 3)) + 1j * rng.standard_normal((7, 3))
    np.testing.assert_allclose(to_real(y) @ real_matrix(W).T, to_real(y @ W.T))


# ---- the federated estimation problem ----------------------------------------

def test_linear_estimation_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    prob = LinearEstimation([rng.standard_normal((9, 4)) for _ in range(3)],
                            [rng.standard_normal((9, 5)) for _ in range(3)])
    assert prob.d == 20
    x = rng.standard_normal(prob.d)
    eps, fd = 1e-6, np.zeros(prob.d)
    for k in range(prob.d):
        e = np.zeros(prob.d)
        e[k] = eps
        fd[k] = (prob.loss(x + e) - prob.loss(x - e)) / (2 * eps)
    np.testing.assert_allclose(prob.grad(x), fd, rtol=1e-5, atol=1e-7)
    assert np.linalg.norm(prob.grad(prob.solution())) < 1e-10


def test_federated_training_equals_pooled_training():
    prob = estimation_problem(make_dataset(ChannelConfig(**SMALL)))
    gamma = 1.0 / prob.smoothness()
    fed = run(prob, GD(), gamma, 100)
    cen = run(prob.pooled(), GD(), gamma, 100)
    np.testing.assert_allclose(fed.x, cen.x, rtol=1e-9, atol=1e-11)


def test_gd_converges_to_the_exact_solution():
    prob = estimation_problem(make_dataset(ChannelConfig(**SMALL)))
    hist = run(prob, GD(), 1.0 / prob.smoothness(), 1500)
    np.testing.assert_allclose(hist.x, prob.solution(), atol=1e-6)


def test_loss_is_half_the_training_error_per_sample():
    ds = make_dataset(ChannelConfig(**SMALL))
    prob = estimation_problem(ds)
    W = lmmse_estimator(ds.P, ds.R_mix, ds.cfg.sigma2)
    x = real_matrix(W).ravel()
    Y, H = pooled(ds.train)
    per_sample = np.sum(np.abs(Y @ W.T - H) ** 2) / len(H)
    assert prob.loss(x) == pytest.approx(per_sample / 2)


def test_learned_estimator_approaches_the_shared_lmmse():
    """With enough training data, the trained matrix is as good as the LMMSE
    estimator built from the true covariance."""
    cfg = ChannelConfig(n_antennas=16, n_pilots=8, n_users=6, m_train=3000, m_test=4000)
    ds = make_dataset(cfg)
    prob = estimation_problem(ds)
    learned = test_nmse_metric(ds, prob)(prob.solution())
    Y, H = pooled(ds.test)
    shared = nmse(lmmse_estimator(ds.P, ds.R_mix, cfg.sigma2), Y, H)
    assert to_db(learned) == pytest.approx(to_db(shared), abs=0.15)
    assert learned < nmse(ls_estimator(ds.P), Y, H)


def test_test_metric_agrees_with_complex_nmse():
    ds = make_dataset(ChannelConfig(**SMALL))
    prob = estimation_problem(ds)
    W = lmmse_estimator(ds.P, ds.R_mix, ds.cfg.sigma2)
    x = real_matrix(W).ravel()
    Y, H = pooled(ds.test)
    assert test_nmse_metric(ds, prob)(x) == pytest.approx(nmse(W, Y, H))
    assert test_nmse_metric(ds, prob, user=2)(x) == pytest.approx(nmse(W, *ds.test[2]))


def test_compressed_training_improves_the_estimator():
    ds = make_dataset(ChannelConfig(**SMALL))
    prob = estimation_problem(ds)
    metric = test_nmse_metric(ds, prob)
    hist = run(prob, EF21(TopK(prob.d // 20)), 0.5 / prob.smoothness(), 600,
               metrics={"nmse": metric}, log_every=100)
    assert hist.extra["nmse"][0] == pytest.approx(1.0)        # W = 0 predicts h = 0
    assert hist.extra["nmse"][-1] < 1.02 * metric(prob.solution())
