"""Tests de métricas, buy & hold y política aleatoria."""
import numpy as np
import pytest

from rl_trader.baselines import buy_and_hold_factory, random_policy_factory
from rl_trader.evaluate import (EpisodeResult, buy_and_hold_final, compute_metrics,
                                evaluate_policy, run_episode, summarize)
from rl_trader.prices import make_series


def _result(balances, positions, rewards=None):
    n = len(positions)
    return EpisodeResult(np.zeros(n + 1), np.zeros(n, dtype=int), np.array(positions),
                         np.array(rewards if rewards is not None else [0.0] * n),
                         np.array(balances, dtype=float))


def test_buy_and_hold_formula():
    p = make_series("browniano", 3, 500)
    res = run_episode(p, buy_and_hold_factory(0))
    m = compute_metrics(res, buy_and_hold_final(p))
    esperado = 1000 * 0.999 / p[5] * p[-1]
    assert buy_and_hold_final(p) == pytest.approx(esperado)
    assert m["final_balance"] == pytest.approx(esperado)
    assert m["n_trades"] == 1
    assert m["time_in_market"] == 1
    assert m["excess_return_pct"] == pytest.approx(0.0, abs=1e-9)


def test_metricas_a_mano():
    res = _result([1000, 1100, 990, 1089], [1, 1, 0])
    m = compute_metrics(res, 1000.0)
    assert m["final_balance"] == 1089
    assert m["log_return"] == pytest.approx(np.log(1.089))
    assert m["return_pct"] == pytest.approx(8.9)
    assert m["excess_return_pct"] == pytest.approx(8.9)
    # pico 1100, valle 990
    assert m["max_drawdown"] == pytest.approx(0.1)
    r = np.array([0.1, -0.1, 0.1])
    assert m["sharpe"] == pytest.approx(r.mean() / r.std())
    assert m["n_trades"] == 2
    assert m["time_in_market"] == pytest.approx(2 / 3)


def test_sharpe_cero_si_std_cero():
    m = compute_metrics(_result([1000, 1000, 1000, 1000], [0, 0, 0]), 1000.0)
    assert m["sharpe"] == 0.0
    assert m["max_drawdown"] == 0.0
    assert m["n_trades"] == 0
    assert m["log_return"] == 0.0


def test_total_reward_suma():
    res = _result([1000, 1010, 1020], [1, 1], rewards=[0.01, 0.02])
    assert compute_metrics(res, 1000.0)["total_reward"] == pytest.approx(0.03)


def test_random_reproducible_por_seed():
    f = random_policy_factory(5)
    p = make_series("browniano", 9, 100)
    a = run_episode(p, f(9))
    b = run_episode(p, f(9))
    np.testing.assert_array_equal(a.actions, b.actions)
    assert not np.array_equal(a.actions, run_episode(p, f(10)).actions)
    assert not np.array_equal(a.actions, run_episode(p, random_policy_factory(6)(9)).actions)


def test_evaluate_policy_filas_y_summarize():
    df = evaluate_policy(random_policy_factory(0), "browniano", "test", 3, 100)
    assert len(df) == 3 and "seed" in df.columns
    df2 = evaluate_policy(random_policy_factory(0), "browniano", "test", 3, 100)
    assert df.equals(df2)
    s = summarize(df)
    assert list(s.columns) == ["mean", "median", "std"]
    assert "seed" not in s.index


def test_evaluate_bh_sin_exceso():
    df = evaluate_policy(buy_and_hold_factory, "browniano", "test", 2, 100)
    assert (df["n_trades"] == 1).all()
    np.testing.assert_allclose(df["excess_return_pct"], 0.0, atol=1e-9)
