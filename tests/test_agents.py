"""Tests de agentes tabulares y política objetivo."""
import numpy as np
import pytest

from rl_trader.agents import QLearning, Sarsa, epsilon_greedy, target_position


def test_epsilon_cero_devuelve_argmax():
    Q = np.zeros((2, 2, 3))
    Q[0, 1] = [0.1, 0.9, 0.3]
    rng = np.random.default_rng(0)
    assert all(epsilon_greedy(Q, (0, 1), 0.0, rng) == 1 for _ in range(50))


def test_epsilon_uno_reparte_acciones():
    Q = np.zeros((1, 2, 3))
    Q[0, 0] = [0.0, 0.0, 5.0]
    rng = np.random.default_rng(0)
    acts = [epsilon_greedy(Q, (0, 0), 1.0, rng) for _ in range(600)]
    assert (np.bincount(acts, minlength=3) > 140).all()


def test_desempate_aleatorio():
    Q = np.zeros((1, 2, 3))
    Q[0, 0] = [1.0, 1.0, 0.0]
    rng = np.random.default_rng(0)
    acts = {epsilon_greedy(Q, (0, 0), 0.0, rng) for _ in range(100)}
    assert acts == {0, 1}


def test_greedy_action_primer_maximo():
    ag = QLearning(2)
    assert ag.greedy_action((0, 0)) == 0
    ag.Q[1, 1] = [0, 2, 2]
    assert ag.greedy_action((1, 1)) == 1


def test_qlearning_update_sin_done():
    ag = QLearning(3, alpha=0.5, gamma=0.9)
    ag.Q[0, 0, 2] = 1.0
    ag.Q[1, 1] = [0.0, 4.0, 2.0]
    ag.update((0, 0), 2, 1.0, (1, 1), False)
    # 1 + 0.5 * (1 + 0.9 * 4 - 1)
    assert ag.Q[0, 0, 2] == pytest.approx(2.8)


def test_qlearning_update_con_done():
    ag = QLearning(3, alpha=0.5, gamma=0.9)
    ag.Q[0, 0, 2] = 1.0
    ag.Q[1, 1] = [0.0, 4.0, 2.0]
    ag.update((0, 0), 2, 3.0, (1, 1), True)
    # 1 + 0.5 * (3 - 1)
    assert ag.Q[0, 0, 2] == pytest.approx(2.0)


def test_sarsa_update_sin_done():
    ag = Sarsa(3, alpha=0.5, gamma=0.9)
    ag.Q[0, 0, 2] = 1.0
    ag.Q[1, 1] = [0.0, 4.0, 2.0]
    ag.update((0, 0), 2, 1.0, (1, 1), 2, False)
    # 1 + 0.5 * (1 + 0.9 * 2 - 1)
    assert ag.Q[0, 0, 2] == pytest.approx(1.9)


def test_sarsa_update_con_done():
    ag = Sarsa(3, alpha=0.5, gamma=0.9)
    ag.Q[0, 0, 2] = 1.0
    ag.Q[1, 1] = [0.0, 4.0, 2.0]
    ag.update((0, 0), 2, 3.0, (1, 1), 1, True)
    assert ag.Q[0, 0, 2] == pytest.approx(2.0)


def test_update_solo_toca_una_celda():
    ag = QLearning(3)
    ag.update((0, 0), 1, 1.0, (1, 1), False)
    assert np.count_nonzero(ag.Q) == 1


def test_decay_epsilon_con_suelo():
    ag = QLearning(2, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.5)
    ag.decay_epsilon()
    assert ag.epsilon == pytest.approx(0.5)
    for _ in range(20):
        ag.decay_epsilon()
    assert ag.epsilon == 0.05


def test_decay_alpha_con_suelo():
    ag = Sarsa(2, alpha=0.1, alpha_decay=0.5, alpha_min=0.01)
    ag.decay_alpha()
    assert ag.alpha == pytest.approx(0.05)
    for _ in range(20):
        ag.decay_alpha()
    assert ag.alpha == 0.01


def test_target_position_casos_equivalentes():
    Q = np.zeros((6, 2, 3))
    # pos 0: argmax 0 o 1 -> 0; argmax 2 -> 1
    Q[0, 0] = [5, 0, 0]
    Q[1, 0] = [0, 5, 0]
    Q[2, 0] = [0, 0, 5]
    # pos 1: argmax 0 -> 0; argmax 1 o 2 -> 1
    Q[3, 1] = [5, 0, 0]
    Q[4, 1] = [0, 5, 0]
    Q[5, 1] = [0, 0, 5]
    t = target_position(Q)
    assert t.shape == (6, 2)
    assert t[0, 0] == 0 and t[1, 0] == 0 and t[2, 0] == 1
    assert t[3, 1] == 0 and t[4, 1] == 1 and t[5, 1] == 1


def test_target_position_q_cero():
    # argmax 0: sin activo se queda fuera, con activo vende
    assert (target_position(np.zeros((2, 2, 3))) == 0).all()
