"""Ejecución de episodios, métricas y evaluación sobre series de test."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from rl_trader.env import SimpleTradingEnv
from rl_trader.prices import make_series, split_seeds

INITIAL_BALANCE = 1000.0
COMMISSION = 0.001
Policy = Callable[[np.ndarray, SimpleTradingEnv], int]


@dataclass
class EpisodeResult:
    prices: np.ndarray
    actions: np.ndarray
    positions: np.ndarray  # posición tras actuar
    rewards: np.ndarray
    balances: np.ndarray   # len pasos + 1, empieza en 1000


def run_episode(prices: np.ndarray, policy: Policy) -> EpisodeResult:
    """Recorre el entorno con la política hasta terminar."""
    env = SimpleTradingEnv(prices)
    obs, _ = env.reset()
    actions, positions, rewards = [], [], []
    balances = [env.balance]
    terminated = False
    while not terminated:
        action = policy(obs, env)
        obs, reward, terminated, _, info = env.step(action)
        actions.append(action)
        positions.append(env.position)
        rewards.append(reward)
        balances.append(info["balance"])
    return EpisodeResult(prices, np.array(actions), np.array(positions),
                         np.array(rewards), np.array(balances))


def buy_and_hold_final(prices: np.ndarray) -> float:
    """Balance final de B&H (fórmula del enunciado): compra en prices[5] con comisión."""
    return INITIAL_BALANCE * (1 - COMMISSION) / prices[5] * prices[-1]


def compute_metrics(result: EpisodeResult, bh_final_balance: float) -> dict[str, float]:
    """Métricas de un episodio; exceso en puntos porcentuales frente a B&H."""
    bal = result.balances
    final = float(bal[-1])
    ret_pct = (final / INITIAL_BALANCE - 1) * 100
    bh_ret_pct = (bh_final_balance / INITIAL_BALANCE - 1) * 100
    step_ret = np.diff(bal) / bal[:-1]
    std = step_ret.std()
    sharpe = float(step_ret.mean() / std) if std > 0 else 0.0
    running_max = np.maximum.accumulate(bal)
    max_dd = float(np.max((running_max - bal) / running_max))
    pos_prev = np.concatenate(([0], result.positions[:-1]))
    return {
        "final_balance": final,
        "log_return": float(np.log(final / INITIAL_BALANCE)),
        "return_pct": float(ret_pct),
        "excess_return_pct": float(ret_pct - bh_ret_pct),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": float(np.sum(result.positions != pos_prev)),
        "time_in_market": float(np.mean(result.positions)),
        "total_reward": float(np.sum(result.rewards)),
    }


def evaluate_policy(policy_factory: Callable[[int], Policy], generator: str,
                    split: str = "test", n_series: int = 100,
                    n_steps: int = 500) -> pd.DataFrame:
    """Una fila por serie (seed + métricas); policy_factory recibe la semilla de la serie."""
    rows = []
    for seed in split_seeds(split, n_series):
        prices = make_series(generator, seed, n_steps)
        result = run_episode(prices, policy_factory(seed))
        metrics = compute_metrics(result, buy_and_hold_final(prices))
        rows.append({"seed": seed, **metrics})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Media, mediana y desviación típica de cada métrica."""
    metrics = df.drop(columns=[c for c in ("seed", "run") if c in df.columns])
    return metrics.agg(["mean", "median", "std"]).T
