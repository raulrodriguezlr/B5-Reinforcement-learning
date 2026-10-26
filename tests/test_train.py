"""Tests de entrenamiento y criterios de convergencia."""
import numpy as np
import pytest

from rl_trader.discretize import fit_discretizer
from rl_trader.prices import make_series, split_seeds
from rl_trader.train import (ma_convergence_episode, policy_convergence_episode,
                             train, val_convergence_episode)


@pytest.fixture(scope="module")
def disc():
    return fit_discretizer("browniano", n_series=10, n_steps=100, n_bins=5)


def _train(disc, **kw):
    return train("qlearning", "browniano", disc, 30, 0, n_steps=100, **kw)


def _val(n):
    return [make_series("browniano", s, 100) for s in split_seeds("val", n)]


# policy_convergence_episode

def test_policy_conv_constante():
    snaps = np.zeros((300, 4, 2), dtype=np.uint8)
    assert policy_convergence_episode(snaps, window=200) == 0


def test_policy_conv_cambio_y_ventana():
    snaps = np.zeros((300, 4, 2), dtype=np.uint8)
    snaps[:50] = 1
    assert policy_convergence_episode(snaps, window=200) == 50
    assert policy_convergence_episode(snaps, window=251) is None


def test_policy_conv_ultimo_punto():
    snaps = np.zeros((10, 4, 2), dtype=np.uint8)
    snaps[:9] = 1
    assert policy_convergence_episode(snaps, window=1) == 9
    assert policy_convergence_episode(snaps, window=2) is None


def test_policy_conv_umbral_agreement():
    snaps = np.zeros((10, 20, 2), dtype=np.uint8)
    snaps[:5, 0, 0] = 1  # 1 de 40 celdas distinta: 97.5 %
    assert policy_convergence_episode(snaps, window=1, agreement=0.95) == 0
    assert policy_convergence_episode(snaps, window=1, agreement=0.99) == 5


# ma_convergence_episode

def test_ma_conv_vacio_y_corto():
    assert ma_convergence_episode(np.array([]), window=5) is None
    assert ma_convergence_episode(np.ones(4), window=5) is None


def test_ma_conv_constante():
    # fin de la primera ventana
    assert ma_convergence_episode(np.ones(50), window=10) == 9


def test_ma_conv_escalon():
    r = np.concatenate([np.zeros(100), np.ones(300)])
    ep = ma_convergence_episode(r, window=10, tol_frac=0.1)
    assert ep is not None and 100 <= ep <= 110


def test_ma_conv_ventana_igual_a_longitud():
    assert ma_convergence_episode(np.arange(5.0), window=5) == 4


# val_convergence_episode

def test_val_conv_vacio():
    assert val_convergence_episode(np.array([]), np.array([]), 100) is None


def test_val_conv_estable_desde_inicio():
    eps = np.array([49, 99, 149, 199])
    vals = np.array([0.5, 0.5, 0.5, 0.5])
    assert val_convergence_episode(eps, vals, 200, window=100) == 49


def test_val_conv_escalon():
    eps = np.arange(49, 1000, 50)
    vals = np.where(eps < 400, 0.0, 1.0)
    # primer punto estable: 449
    assert val_convergence_episode(eps, vals, 1000, window=200) == 449
    assert val_convergence_episode(eps, vals, 1000, window=600) is None


def test_val_conv_ultimo_punto():
    eps = np.array([49, 99, 149])
    vals = np.array([0.0, 0.0, 1.0])
    assert val_convergence_episode(eps, vals, 150, window=0) == 149
    assert val_convergence_episode(eps, vals, 150, window=1) is None


def test_val_conv_tolerancia_absoluta():
    eps = np.array([49, 99])
    vals = np.array([0.005, 0.0])
    assert val_convergence_episode(eps, vals, 100, tol_abs=0.01, window=0) == 49


# train

def test_train_corto_reproducible(disc):
    a, b = _train(disc), _train(disc)
    np.testing.assert_array_equal(a.Q, b.Q)
    np.testing.assert_array_equal(a.episode_rewards, b.episode_rewards)
    np.testing.assert_array_equal(a.policy_snapshots, b.policy_snapshots)
    assert a.Q.shape == (5, 2, 3)
    assert a.policy_snapshots.shape == (30, 5, 2)
    assert len(a.cumulative_time) == 30
    assert (np.diff(a.epsilons) <= 0).all()


def test_train_sarsa_reproducible(disc):
    a = train("sarsa", "browniano", disc, 20, 1, n_steps=100)
    b = train("sarsa", "browniano", disc, 20, 1, n_steps=100)
    np.testing.assert_array_equal(a.Q, b.Q)


def test_train_distinto_run_distinta_q(disc):
    a = _train(disc)
    b = train("qlearning", "browniano", disc, 30, 1, n_steps=100)
    assert not np.array_equal(a.Q, b.Q)


def test_train_identico_con_y_sin_val(disc):
    sin = _train(disc)
    con = _train(disc, val_series=_val(2), val_every=10)
    np.testing.assert_array_equal(sin.Q, con.Q)
    np.testing.assert_array_equal(sin.episode_rewards, con.episode_rewards)
    np.testing.assert_array_equal(sin.policy_snapshots, con.policy_snapshots)
    assert len(sin.val_episodes) == 0 and sin.convergence_episode_val is None
    assert list(con.val_episodes) == [9, 19, 29]
    assert len(con.val_median_log_return) == 3


def test_train_val_every_cero_con_val(disc):
    with pytest.raises(ValueError):
        _train(disc, val_series=_val(1), val_every=0)


def test_train_val_every_cero_sin_val_ok(disc):
    assert len(_train(disc, val_every=0).val_episodes) == 0


def test_val_episodes_sin_duplicado_n_no_multiplo(disc):
    r = train("qlearning", "browniano", disc, 25, 0, n_steps=100,
              val_series=_val(2), val_every=10)
    assert list(r.val_episodes) == [9, 19, 24]
    # n múltiplo: el último punto no se repite
    r2 = train("qlearning", "browniano", disc, 20, 0, n_steps=100,
               val_series=_val(2), val_every=10)
    assert list(r2.val_episodes) == [9, 19]
