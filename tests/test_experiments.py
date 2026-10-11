"""Tests for step 6: the helpers that keep comparisons fair.

Run with:  pytest -q
"""

import numpy as np
import pytest

from fedcomp import dense_bits, randk_bits, topk_bits
from fedcomp.channel import ChannelConfig, estimation_problem, make_dataset, test_nmse_metric
from fedcomp.experiments import (METHODS, bits_per_round, bits_to_reach, make_method,
                                 repeat_run, tuned_run)


def small_task():
    ds = make_dataset(ChannelConfig(n_antennas=16, n_pilots=8, n_users=6, m_train=300, m_test=1000))
    problem = estimation_problem(ds)
    metric = test_nmse_metric(ds, problem)
    return problem, metric, metric(problem.solution())


def test_bits_to_reach():
    bits, err = [0, 10, 20, 30], [1.0, 0.5, 0.2, 0.1]
    assert bits_to_reach(bits, err, 0.5) == 10
    assert bits_to_reach(bits, err, 0.15) == 30
    assert bits_to_reach(bits, err, 0.05) is None


def test_bits_per_round_by_family():
    assert bits_per_round("gd", 5, 100) == dense_bits(100)
    assert bits_per_round("ef21_top", 5, 100) == topk_bits(5, 100)
    assert bits_per_round("diana_rand", 5, 100) == randk_bits(5)


def test_every_method_can_be_built_and_named():
    for key in METHODS:
        method = make_method(key, 4)
        method.setup(3, 20, np.ones(3) / 3)
        assert method.name


def test_every_method_gets_the_same_bit_budget():
    problem, metric, best = small_task()
    budget = 20 * dense_bits(problem.d)
    for key in ("gd", "ef21_top", "diana_rand"):
        res = tuned_run(problem, key, problem.d // 10, budget, metric, 1.02 * best,
                        multipliers=(0.25, 1.0))
        assert res["bits"][-1] <= budget
        assert res["bits"][-1] > budget - bits_per_round(key, problem.d // 10, problem.d)


def test_tuning_picks_the_stepsize_that_reaches_the_target_first():
    problem, metric, best = small_task()
    budget, target = 40 * dense_bits(problem.d), 1.05 * best
    grid = (0.03125, 0.25, 1.0)
    chosen = tuned_run(problem, "gd", 1, budget, metric, target, multipliers=grid)
    alone = [tuned_run(problem, "gd", 1, budget, metric, target, multipliers=(m,)) for m in grid]
    reached = [r["bits_to_target"] for r in alone if r["bits_to_target"] is not None]
    assert chosen["bits_to_target"] == min(reached)
    assert chosen["multiplier"] == 1.0


def test_diverging_stepsizes_are_discarded():
    problem, metric, best = small_task()
    res = tuned_run(problem, "gd", 1, 40 * dense_bits(problem.d), metric, 1.05 * best,
                    multipliers=(1.0, 64.0))
    assert res["multiplier"] == 1.0


def test_repeat_run_averages_over_seeds():
    problem, metric, best = small_task()
    budget = 10 * dense_bits(problem.d)
    rep = repeat_run(problem, "diana_rand", problem.d // 10, budget, metric, 0.25, seeds=[0, 1, 2])
    err, lo, hi = map(np.array, (rep["err"], rep["err_min"], rep["err_max"]))
    assert np.all(lo <= err + 1e-12) and np.all(err <= hi + 1e-12)
    assert np.any(hi - lo > 0)            # the seeds really differ
    assert err[0] == pytest.approx(1.0)
