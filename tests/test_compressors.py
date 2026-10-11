"""Tests for step 3: the compressors and their bit costs.

Run with:  pytest -q
"""

from itertools import combinations

import numpy as np
import pytest

from fedcomp import Identity, RandK, TopK, dense_bits, index_bits, randk_bits, topk_bits

# the example vector used in the study notes (Part 2)
G = np.array([0.8, -0.3, 0.05, 1.2, -0.6])
G_NORM_SQ = float(G @ G)  # 2.5325


def ready(comp, d):
    comp.setup(d)
    return comp


def all_randk_outputs(x, k, scaled=True):
    """Every possible Rand-k output, each equally likely."""
    d = len(x)
    outs = []
    for idx in combinations(range(d), k):
        out = np.zeros(d)
        out[list(idx)] = x[list(idx)] * (d / k if scaled else 1.0)
        outs.append(out)
    return np.array(outs)


# ---- the worked examples from the notes -----------------------------------

def test_topk_on_notes_vector():
    out, bits = ready(TopK(2), 5)(G)
    np.testing.assert_allclose(out, [0.8, 0, 0, 1.2, 0])
    assert float((out - G) @ (out - G)) == pytest.approx(0.4525)
    assert bits == 70


def test_randk_on_notes_vector():
    out, bits = ready(RandK(2), 5)(G, np.random.default_rng(0))
    kept = np.flatnonzero(out)
    assert len(kept) == 2
    np.testing.assert_allclose(out[kept], 2.5 * G[kept])
    assert bits == 96


# ---- Rand-k: exact properties, by listing every possible mask --------------

@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_randk_is_exactly_unbiased(k):
    outs = all_randk_outputs(G, k)
    np.testing.assert_allclose(outs.mean(axis=0), G, atol=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_randk_variance_formula_is_exact(k):
    outs = all_randk_outputs(G, k)
    variance = np.mean(np.sum((outs - G) ** 2, axis=1))
    comp = ready(RandK(k), 5)
    assert variance == pytest.approx(comp.omega * G_NORM_SQ)
    assert comp.omega == pytest.approx(5 / k - 1)


def test_randk_variance_on_notes_vector_is_3_79875():
    outs = all_randk_outputs(G, 2)
    assert np.mean(np.sum((outs - G) ** 2, axis=1)) == pytest.approx(3.79875)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_unscaled_randk_is_contractive_with_alpha_k_over_d(k):
    outs = all_randk_outputs(G, k, scaled=False)
    err = np.mean(np.sum((outs - G) ** 2, axis=1))
    comp = ready(RandK(k, scaled=False), 5)
    assert comp.alpha == pytest.approx(k / 5)
    assert err == pytest.approx((1 - comp.alpha) * G_NORM_SQ)
    assert not np.allclose(outs.mean(axis=0), G)  # and it is biased


# ---- Rand-k: the sampler itself, statistically ------------------------------

def test_randk_sampler_matches_theory():
    d, k, draws = 40, 5, 40000
    rng = np.random.default_rng(1)
    x = rng.standard_normal(d)
    comp = ready(RandK(k), d)
    outs = np.array([comp(x, rng)[0] for _ in range(draws)])

    assert np.all(np.count_nonzero(outs, axis=1) <= k)
    # mean close to x: each coordinate's standard error is |x_j| sqrt(omega/draws)
    tol = 6 * np.abs(x) * np.sqrt(comp.omega / draws) + 1e-12
    assert np.all(np.abs(outs.mean(axis=0) - x) <= tol)
    # average squared error close to omega ||x||^2
    err = np.mean(np.sum((outs - x) ** 2, axis=1))
    assert err == pytest.approx(comp.omega * float(x @ x), rel=0.03)


def test_randk_picks_every_coordinate_equally_often():
    d, k, draws = 10, 3, 30000
    rng = np.random.default_rng(2)
    comp = ready(RandK(k), d)
    counts = np.zeros(d)
    for _ in range(draws):
        counts += comp(np.ones(d), rng)[0] != 0
    np.testing.assert_allclose(counts / draws, k / d, atol=0.015)


def test_randk_is_reproducible_from_its_seed():
    comp = ready(RandK(3), 20)
    x = np.arange(20.0)
    a = comp(x, np.random.default_rng(7))[0]
    b = comp(x, np.random.default_rng(7))[0]   # what the server regenerates
    c = comp(x, np.random.default_rng(8))[0]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ---- Top-k -----------------------------------------------------------------

def test_topk_keeps_the_largest_magnitudes():
    x = np.array([3.0, -7.0, 0.5, 6.0, -1.0, 2.0])
    out, _ = ready(TopK(3), 6)(x)
    np.testing.assert_allclose(out, [3.0, -7.0, 0, 6.0, 0, 0])


def test_topk_contraction_bound_holds():
    rng = np.random.default_rng(3)
    for d, k in [(10, 1), (10, 3), (50, 5), (50, 49)]:
        comp = ready(TopK(k), d)
        for _ in range(200):
            x = rng.standard_normal(d) * rng.uniform(0.1, 10, size=d)
            out, _ = comp(x)
            assert np.sum((out - x) ** 2) <= (1 - comp.alpha) * (x @ x) * (1 + 1e-12)


def test_topk_bound_is_tight_for_equal_magnitudes():
    x = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    comp = ready(TopK(2), 8)
    out, _ = comp(x)
    assert np.sum((out - x) ** 2) == pytest.approx((1 - comp.alpha) * (x @ x))


def test_topk_is_biased_and_has_lower_error_than_randk():
    out, _ = ready(TopK(2), 5)(G)
    assert not np.allclose(out, G)
    topk_err = np.sum((out - G) ** 2)
    randk_err = np.mean(np.sum((all_randk_outputs(G, 2) - G) ** 2, axis=1))
    assert topk_err < randk_err


# ---- edge cases ------------------------------------------------------------

@pytest.mark.parametrize("comp", [RandK(5), RandK(5, scaled=False), TopK(5), Identity()])
def test_k_equal_d_sends_everything(comp):
    out, _ = ready(comp, 5)(G, np.random.default_rng(0))
    np.testing.assert_allclose(out, G)


@pytest.mark.parametrize("make", [RandK, TopK])
def test_invalid_k_is_rejected(make):
    with pytest.raises(ValueError):
        make(0)
    with pytest.raises(ValueError):
        make(6).setup(5)


@pytest.mark.parametrize("comp", [RandK(2), TopK(2), Identity()])
def test_input_is_not_modified(comp):
    x = G.copy()
    ready(comp, 5)(x, np.random.default_rng(0))
    np.testing.assert_array_equal(x, G)


def test_class_flags():
    assert ready(RandK(2), 5).unbiased
    assert not ready(TopK(2), 5).unbiased
    assert not ready(RandK(2, scaled=False), 5).unbiased
    assert ready(Identity(), 5).unbiased and Identity.alpha == 1.0


# ---- bit costs -------------------------------------------------------------

def test_bit_costs_small():
    assert dense_bits(5) == 160
    assert index_bits(5) == 3
    assert randk_bits(2) == 96
    assert topk_bits(2, 5) == 70


def test_bit_costs_at_one_million_parameters():
    d, k = 1_000_000, 10_000
    assert dense_bits(d) == 32_000_000
    assert index_bits(d) == 20
    assert randk_bits(k) == 320_032
    assert topk_bits(k, d) == 520_000


def test_compressors_report_their_bit_costs():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    assert ready(RandK(10), 1000)(x, rng)[1] == randk_bits(10)
    assert ready(TopK(10), 1000)(x)[1] == topk_bits(10, 1000)
    assert ready(Identity(), 1000)(x)[1] == dense_bits(1000)
