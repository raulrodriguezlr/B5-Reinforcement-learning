"""Tests de las extensiones: variante de comisión, Monte Carlo y entrenamiento mixto."""
import copy

import numpy as np
import pytest

from rl_trader.agents import MonteCarlo, MonteCarloOff
from rl_trader.env import SimpleTradingEnv
from rl_trader.env_variants import CostTradingEnv, make_env
from rl_trader.experiments import run_mixed
from rl_trader.prices import make_series
from rl_trader.train import train
from rl_trader.discretize import fit_discretizer
from test_experiments import CFG as BASE


def _rollout(env, actions):
    env.reset()
    out = []
    for a in actions:
        obs, r, done, _, info = env.step(a)
        out.append((r, info["balance"]))
        if done:
            break
    return out


def test_make_env_base_es_el_del_enunciado():
    prices = make_series("browniano", 1, 50)
    assert type(make_env(prices)) is SimpleTradingEnv
    assert type(make_env(prices, 0.005)) is CostTradingEnv


def test_variante_con_comision_base_coincide_con_el_entorno_base():
    prices = make_series("browniano", 2, 60)
    acts = [2, 1, 1, 0, 2, 1, 0, 0, 2] * 5
    assert _rollout(CostTradingEnv(prices, 0.001), acts) == _rollout(SimpleTradingEnv(prices), acts)


def test_comision_mayor_reduce_balance_y_recompensa():
    prices = make_series("browniano", 3, 40)
    acts = [2, 0] * 10
    lo, hi = _rollout(CostTradingEnv(prices, 0.0), acts), _rollout(CostTradingEnv(prices, 0.01), acts)
    assert hi[-1][1] < lo[-1][1]
    assert sum(r for r, _ in hi) < sum(r for r, _ in lo)


def test_mc_primera_visita_promedia_retornos():
    ag = MonteCarlo(2, gamma=0.5)
    s, s2 = (0, 0), (1, 0)
    # (s,a=2) aparece dos veces: solo cuenta la primera visita; G0 = 1 + 0.5*(0 + 0.5*2)
    ag.update_episode([(s, 2, 1.0), (s2, 1, 0.0), (s, 2, 2.0)])
    assert ag.Q[s][2] == pytest.approx(1.5)
    assert ag.N[s][2] == 1
    assert ag.Q[s2][1] == pytest.approx(1.0)


def test_mc_off_corta_si_la_accion_no_es_greedy():
    ag = MonteCarloOff(2, gamma=1.0, epsilon_start=0.5)
    s = (0, 0)
    ag.Q[s] = np.array([0.0, 5.0, 0.0])  # greedy = 1
    ag.update_episode([(s, 2, 1.0), (s, 1, 1.0)])
    # último paso (a=1, greedy) se actualiza y sigue; el primero (a=2) se actualiza y corta
    assert ag.C[s][1] > 0 and ag.C[s][2] > 0


@pytest.mark.parametrize("algo", ["mc", "mc_off"])
def test_train_con_mc_corre_y_produce_q(algo):
    disc = fit_discretizer("browniano", 5, 60, 4)
    res = train(algo, "browniano", disc, 8, 0, 60)
    assert res.Q.shape == (4, 2, 3)
    assert np.isfinite(res.Q).all()


def test_train_alterna_generadores():
    disc = fit_discretizer("browniano", 5, 60, 4)
    res = train("qlearning", ["browniano", "regimenes"], disc, 6, 0, 60)
    assert len(res.episode_rewards) == 6


def test_run_mixed_filas(tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg["n_runs"] = 1
    out = run_mixed(cfg, tmp_path, ["qlearning"], 6)
    assert set(out["generator"]) == {"tendencia_ciclo", "browniano", "reversion_media", "regimenes"}
    assert (out["method"] == "qlearning_mixto").all()
