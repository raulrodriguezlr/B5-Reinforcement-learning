"""Tests de la discretización por cuantiles."""
import numpy as np
import pytest

from rl_trader.discretize import TrendDiscretizer, fit_discretizer, trend_series
from rl_trader.env import SimpleTradingEnv
from rl_trader.prices import make_series, split_seeds


@pytest.mark.parametrize("gen", ["tendencia_ciclo", "browniano", "regimenes"])
def test_trend_series_igual_a_obs_del_env(gen):
    prices = make_series(gen, 11, 200)
    ts = trend_series(prices)
    env = SimpleTradingEnv(prices)
    obs, _ = env.reset()
    # estados de decisión: t = 5 .. len-2 (el último obs ya es terminal)
    for i in range(len(ts)):
        assert obs[0] == ts[i]
        obs, _, _, _, _ = env.step(1)
    assert len(ts) == len(prices) - 6


def test_trend_series_longitud_y_dtype():
    ts = trend_series(np.linspace(100, 110, 50))
    assert ts.dtype == np.float32 and len(ts) == 44


def test_cortes_crecientes_y_n_bins():
    d = fit_discretizer("browniano", n_series=20, n_steps=200, n_bins=10)
    assert d.n_bins == 10
    assert len(d.edges) == 9
    assert (np.diff(d.edges) > 0).all()


def test_ocupacion_aproximadamente_uniforme_en_fit():
    n_bins = 10
    d = fit_discretizer("browniano", n_series=50, n_steps=300, n_bins=n_bins)
    trends = np.concatenate([trend_series(make_series("browniano", s, 300))
                             for s in split_seeds("fit", 50)])
    bins = np.array([d.bin(t) for t in trends])
    occ = np.bincount(bins, minlength=n_bins) / len(bins)
    np.testing.assert_allclose(occ, 1 / n_bins, atol=0.01)


def test_bins_en_rango_en_extremos():
    d = fit_discretizer("browniano", n_series=10, n_steps=200, n_bins=10)
    assert d.bin(-1.0) == 0
    assert d.bin(1.0) == d.n_bins - 1
    for t in np.linspace(-1, 1, 101):
        assert 0 <= d.bin(float(t)) <= d.n_bins - 1


def test_state_devuelve_bin_y_posicion():
    d = TrendDiscretizer(np.array([-0.1, 0.0, 0.1]))
    assert d.state(np.array([0.05, 1], dtype=np.float32)) == (2, 1)
    assert d.state(np.array([-0.5, 0], dtype=np.float32)) == (0, 0)


def test_dict_ida_y_vuelta():
    d = fit_discretizer("regimenes", n_series=10, n_steps=200, n_bins=8)
    d2 = TrendDiscretizer.from_dict(d.to_dict())
    np.testing.assert_array_equal(d.edges, d2.edges)
    assert d2.n_bins == d.n_bins == 8
    assert d2.to_dict() == d.to_dict()


def test_fit_determinista():
    a = fit_discretizer("browniano", n_series=5, n_steps=100, n_bins=5)
    b = fit_discretizer("browniano", n_series=5, n_steps=100, n_bins=5)
    np.testing.assert_array_equal(a.edges, b.edges)
