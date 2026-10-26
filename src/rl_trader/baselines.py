"""Políticas de referencia: aleatoria y buy & hold."""
from typing import Callable

import numpy as np

from rl_trader.env import SimpleTradingEnv


def random_policy_factory(base_seed: int = 0) -> Callable[[int], Callable]:
    """Factoría de políticas aleatorias sembradas por serie (base_seed + series_seed)."""
    def factory(series_seed: int) -> Callable[[np.ndarray, SimpleTradingEnv], int]:
        seeded = False

        def policy(obs: np.ndarray, env: SimpleTradingEnv) -> int:
            nonlocal seeded
            if not seeded:
                env.action_space.seed(base_seed + series_seed)
                seeded = True
            return int(env.action_space.sample())
        return policy
    return factory


def buy_and_hold_policy(obs: np.ndarray, env: SimpleTradingEnv) -> int:
    """Compra al inicio y mantiene hasta el final."""
    return 2 if env.position == 0 else 1


def buy_and_hold_factory(series_seed: int) -> Callable:
    """Factoría trivial para encajar con evaluate_policy."""
    return buy_and_hold_policy
